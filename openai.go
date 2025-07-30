package echo

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
}

type OpenAIResponse struct {
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
func (c *OpenAIClient) prepareRequest(prompt string, streaming bool, opts ...CallOption) (OpenAIRequest, CallConfig) {
	// Start with client's default call config
	callCfg := *c.cfg
	// Apply call-specific options (these override client defaults)
	for _, opt := range opts {
		opt(&callCfg)
	}

	// Build messages array
	messages := []OpenAIMessage{}

	// Add system message if provided
	if callCfg.SystemMsg != "" {
		messages = append(messages, OpenAIMessage{
			Role:    "system",
			Content: callCfg.SystemMsg,
		})
	}

	// Add user message
	messages = append(messages, OpenAIMessage{
		Role:    "user",
		Content: prompt,
	})

	req := OpenAIRequest{
		Model:       callCfg.Model,
		Temperature: callCfg.Temperature,
		MaxTokens:   callCfg.MaxTokens,
		Messages:    messages,
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

	return req, callCfg
}

func (c *OpenAIClient) Call(ctx context.Context, prompt string, opts ...CallOption) (*Response, error) {
	body, callCfg := c.prepareRequest(prompt, false, opts...)

	resp := OpenAIResponse{}
	err := callHTTPAPI(ctx, callCfg.BaseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API call failed: %w", err)
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


func (c *OpenAIClient) StreamCall(ctx context.Context, prompt string, opts ...CallOption) (*StreamResponse, error) {
	body, callCfg := c.prepareRequest(prompt, true, opts...)

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
