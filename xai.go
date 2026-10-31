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

// xaiRequest represents a request to the xAI chat completions API
type xaiRequest struct {
	Model         string          `json:"model"`
	Temperature   *float32        `json:"temperature,omitempty"`
	MaxTokens     *int            `json:"max_completion_tokens,omitempty"`
	Messages      []openAIMessage `json:"messages"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
	ResponseFormat  *openAIResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort string                `json:"reasoning_effort,omitempty"`
	Store           *bool                 `json:"store,omitempty"` // xAI-specific: set to false to disable server-side storage
}

// xaiError represents an error from the xAI API
type xaiError struct {
	Message string `json:"message"`
	Code    int32  `json:"code"`
}

// xaiResponse represents a response from the xAI chat completions API
type xaiResponse struct {
	Error   *xaiError `json:"error,omitempty"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *openAIUsage `json:"usage,omitempty"`
}

// xaiStreamResponse represents a streaming response chunk from xAI
type xaiStreamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *openAIUsage `json:"usage,omitempty"`
}

// XAIProvider is a stateless provider for xAI (Grok) API
type XAIProvider struct {
	Key string
}

// NewXAIClient creates a new xAI client
func NewXAIClient(apiKey, model string, opts ...CallOption) Client {
	if model != "" {
		if !strings.HasPrefix(model, "xai/") {
			model = "xai/" + model
		}
		opts = append([]CallOption{WithModel(model)}, opts...)
	}
	client, _ := NewClient(opts...)
	client.SetProvider("xai", &XAIProvider{Key: apiKey})
	return client
}

// prepareXAIRequest builds the xAI request with the given configuration
func prepareXAIRequest(messages []Message, streaming bool, cfg CallConfig) (xaiRequest, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return xaiRequest{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Convert messages to OpenAI format (xAI is OpenAI-compatible)
	xaiMessages := []openAIMessage{}
	systemMessageProcessed := false

	for _, msg := range messages {
		switch msg.Role {
		case System:
			// Skip system message here if WithSystemMessage is set
			if cfg.SystemMsg == "" {
				xaiMessages = append(xaiMessages, openAIMessage{
					Role:    "system",
					Content: msg.Content,
				})
			}
			systemMessageProcessed = true
		case User:
			xaiMessages = append(xaiMessages, openAIMessage{
				Role:    "user",
				Content: msg.Content,
			})
		case Agent:
			xaiMessages = append(xaiMessages, openAIMessage{
				Role:    "assistant",
				Content: msg.Content,
			})
		}
	}

	// Handle WithSystemMessage option
	if cfg.SystemMsg != "" {
		// Insert system message at the beginning
		systemMsg := openAIMessage{
			Role:    "system",
			Content: cfg.SystemMsg,
		}
		if systemMessageProcessed {
			// Replace the first message (which should be system)
			xaiMessages = append([]openAIMessage{systemMsg}, xaiMessages[1:]...)
		} else {
			// Add system message at the beginning
			xaiMessages = append([]openAIMessage{systemMsg}, xaiMessages...)
		}
	}

	req := xaiRequest{
		Model:       cfg.Model,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		Messages:    xaiMessages,
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

	// Handle store parameter - default to false for privacy
	if cfg.StoreData != nil {
		req.Store = cfg.StoreData
	} else {
		// Default to false (disable server-side storage)
		store := false
		req.Store = &store
	}

	// Add structured output response format if configured
	if cfg.StructuredOutput != nil {
		req.ResponseFormat = &openAIResponseFormat{
			Type: "json_schema",
			JSONSchema: &openAIJSONSchemaConfig{
				Name:   cfg.StructuredOutput.Name,
				Strict: true,
				Schema: cfg.StructuredOutput.Schema,
			},
		}
	}

	// Add reasoning effort if configured
	if cfg.ReasoningEffort != "" {
		req.ReasoningEffort = cfg.ReasoningEffort
	}

	return req, nil
}

// call implements the provider interface for xAI
func (p *XAIProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
	body, err := prepareXAIRequest(messages, false, cfg)
	if err != nil {
		return nil, err
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.x.ai/v1/chat/completions"
	}

	resp := xaiResponse{}
	err = callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("xAI API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("xAI API error: %s", resp.Error.Message)
	}

	// Extract text from LLM response
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	return &Response{
		Text:  resp.Choices[0].Message.Content,
		Usage: resp.Usage.toUsage(),
	}, nil
}

// streamCall implements the provider interface for xAI streaming
func (p *XAIProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	body, err := prepareXAIRequest(messages, true, cfg)
	if err != nil {
		return nil, err
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.x.ai/v1/chat/completions"
	}

	// Get streaming response
	respBody, err := streamHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}, body)
	if err != nil {
		return nil, fmt.Errorf("xAI streaming API call failed: %w", err)
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
			var streamResp xaiStreamResponse
			if err := json.Unmarshal(data, &streamResp); err != nil {
				ch <- StreamChunk{Error: fmt.Errorf("json parse error: %w", err)}
				return
			}

			if len(streamResp.Choices) > 0 && streamResp.Choices[0].Delta.Content != "" {
				ch <- StreamChunk{
					Data: streamResp.Choices[0].Delta.Content,
				}
			}

			// Usage arrives on the final chunk (some servers also include choices there)
			if streamResp.Usage != nil {
				ch <- StreamChunk{
					Usage: streamResp.Usage.toUsage(),
				}
			}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}
