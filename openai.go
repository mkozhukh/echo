package echo

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenRouterProvider struct {
	Order          []string `json:"order"`
	Only           []string `json:"only"`
	AllowFallbacks bool     `json:"allow_fallbacks"`
}

type OpenAIError struct {
	Message string `json:"message"`
	Code    int32  `json:"code"`
}

type OpenAIRequest struct {
	Model         string          `json:"model"`
	Temperature   *float64        `json:"temperature,omitempty"`
	MaxTokens     *int            `json:"max_tokens,omitempty"`
	Messages      []OpenAIMessage `json:"messages"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
	Provider *OpenRouterProvider `json:"provider,omitempty"`
}

type OpenAIResponse struct {
	Error   *OpenAIError `json:"error,omitempty"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

type OpenAIClient struct {
	apiKey string
	cfg    *CallConfig
}

// Convenience functions for easy client creation
func NewOpenAIClient(apiKey, model string, opts ...CallOption) *OpenAIClient {
	cfg := &CallConfig{
		BaseURL: "https://api.openai.com/v1/chat/completions",
		Model:   model,
	}

	// Apply client options
	for _, opt := range opts {
		opt(cfg)
	}

	return &OpenAIClient{apiKey: apiKey, cfg: cfg}
}

// prepareRequest builds the OpenAI request with the given configuration
func (c *OpenAIClient) prepareRequest(messages []Message, streaming bool, opts ...CallOption) (OpenAIRequest, CallConfig, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return OpenAIRequest{}, CallConfig{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Start with client's default call config
	callCfg := *c.cfg
	// Apply call-specific options (these override client defaults)
	for _, opt := range opts {
		opt(&callCfg)
	}

	// Convert messages to OpenAI format
	openaiMessages := []OpenAIMessage{}
	systemMessageProcessed := false

	for _, msg := range messages {
		switch msg.Role {
		case System:
			// Skip system message here if WithSystemMessage is set
			if callCfg.SystemMsg == "" {
				openaiMessages = append(openaiMessages, OpenAIMessage{
					Role:    "system",
					Content: msg.Content,
				})
			}
			systemMessageProcessed = true
		case User:
			openaiMessages = append(openaiMessages, OpenAIMessage{
				Role:    "user",
				Content: msg.Content,
			})
		case Agent:
			openaiMessages = append(openaiMessages, OpenAIMessage{
				Role:    "assistant",
				Content: msg.Content,
			})
		}
	}

	// Handle WithSystemMessage option
	if callCfg.SystemMsg != "" {
		// Insert system message at the beginning
		systemMsg := OpenAIMessage{
			Role:    "system",
			Content: callCfg.SystemMsg,
		}
		if systemMessageProcessed {
			// Replace the first message (which should be system)
			openaiMessages = append([]OpenAIMessage{systemMsg}, openaiMessages[1:]...)
		} else {
			// Add system message at the beginning
			openaiMessages = append([]OpenAIMessage{systemMsg}, openaiMessages...)
		}
	}

	req := OpenAIRequest{
		Model:       callCfg.Model,
		Temperature: callCfg.Temperature,
		MaxTokens:   callCfg.MaxTokens,
		Messages:    openaiMessages,
		Stream:      streaming,
	}

	// Add stream options for usage stats when streaming
	if streaming {
		req.StreamOptions = &struct {
			IncludeUsage bool `json:"include_usage"`
		}{
			IncludeUsage: true,
		}
	}

	// Add provider field if EndPoint is set (for openrouter compatibility)
	if callCfg.EndPoint != "" {
		order := strings.Split(callCfg.EndPoint, ",")
		req.Provider = &OpenRouterProvider{
			Only:           order,
			Order:          order,
			AllowFallbacks: true,
		}
	}

	return req, callCfg, nil
}

func (c *OpenAIClient) Call(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error) {
	body, callCfg, err := c.prepareRequest(messages, false, opts...)
	if err != nil {
		return nil, err
	}

	resp := OpenAIResponse{}
	err = callHTTPAPI(ctx, callCfg.BaseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("OpenAI API error: %s", resp.Error.Message)
	}

	// Extract text from LLM response
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	response := &Response{
		Text: resp.Choices[0].Message.Content,
	}

	// Add metadata if usage information is available
	if resp.Usage != nil {
		response.Metadata = Metadata{
			"total_tokens":      resp.Usage.TotalTokens,
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
		}
	}

	return response, nil
}

// Streaming response structures
type OpenAIStreamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

func (c *OpenAIClient) StreamCall(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error) {
	body, callCfg, err := c.prepareRequest(messages, true, opts...)
	if err != nil {
		return nil, err
	}

	// Get streaming response
	respBody, err := streamHTTPAPI(ctx, callCfg.BaseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}, body)
	if err != nil {
		return nil, fmt.Errorf("OpenAI streaming API call failed: %w", err)
	}

	// Create channel for streaming
	ch := make(chan StreamChunk)

	// Start goroutine to process stream
	go func() {
		defer close(ch)
		defer respBody.Close()

		reader := bufio.NewReader(respBody)
		for {
			line, err := reader.ReadBytes('\n')
			if err == io.EOF {
				break
			}
			if err != nil {
				ch <- StreamChunk{Error: fmt.Errorf("read error: %w", err)}
				return
			}

			// Skip empty lines
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			// Check for SSE data prefix
			if !bytes.HasPrefix(line, dataPrefix) {
				continue
			}

			// Remove "data: " prefix
			data := bytes.TrimPrefix(line, dataPrefix)

			// Check for end of stream
			if bytes.Equal(data, doneMarker) {
				return
			}

			// Parse JSON
			var streamResp OpenAIStreamResponse
			if err := json.Unmarshal(data, &streamResp); err != nil {
				ch <- StreamChunk{Error: fmt.Errorf("json parse error: %w", err)}
				return
			}

			// Check if this is a usage chunk (has usage data but no choices)
			if streamResp.Usage != nil && len(streamResp.Choices) == 0 {
				// Send metadata chunk
				meta := Metadata{
					"total_tokens":      streamResp.Usage.TotalTokens,
					"prompt_tokens":     streamResp.Usage.PromptTokens,
					"completion_tokens": streamResp.Usage.CompletionTokens,
				}
				ch <- StreamChunk{
					Meta: &meta,
				}
			} else if len(streamResp.Choices) > 0 && streamResp.Choices[0].Delta.Content != "" {
				// Normal content chunk
				ch <- StreamChunk{
					Data: streamResp.Choices[0].Delta.Content,
				}
			}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}
