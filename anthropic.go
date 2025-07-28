package echo

import (
	"context"
	"fmt"
	"net/http"
)

type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type AnthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []AnthropicMessage `json:"messages"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature *float64           `json:"temperature,omitempty"`
	System      string             `json:"system,omitempty"`
}

type AnthropicResponse struct {
	Content []struct {
		Text string `json:"text"`
		Type string `json:"type"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type AnthropicClient struct {
	apiKey string
	cfg    *CallConfig
}

// NewAnthropicClient creates a new Anthropic client with full configuration
func NewAnthropicClient(apiKey, model string, opts ...CallOption) *AnthropicClient {
	cfg := &CallConfig{
		BaseURL: "https://api.anthropic.com/v1/messages",
		Model:   model,
	}

	// Apply client options
	for _, opt := range opts {
		opt(cfg)
	}

	return &AnthropicClient{apiKey: apiKey, cfg: cfg}
}

// Convenience functions for common models
func NewClaude4Opus(apiKey string, opts ...CallOption) *AnthropicClient {
	return NewAnthropicClient(apiKey, "claude-opus-4-20250514", opts...)
}

func NewClaude4Sonnet(apiKey string, opts ...CallOption) *AnthropicClient {
	return NewAnthropicClient(apiKey, "claude-sonnet-4-20250514", opts...)
}

func NewClaude35Haiku(apiKey string, opts ...CallOption) *AnthropicClient {
	return NewAnthropicClient(apiKey, "claude-3-5-haiku-20241022", opts...)
}

func (c *AnthropicClient) Call(ctx context.Context, prompt string, opts ...CallOption) (*Response, error) {
	// Start with client's default call config
	callCfg := *c.cfg
	// Apply call-specific options (these override client defaults)
	for _, opt := range opts {
		opt(&callCfg)
	}

	// Build messages array
	messages := []AnthropicMessage{
		{
			Role:    "user",
			Content: prompt,
		},
	}

	// Anthropic requires max_tokens to be set
	maxTokens := 4096
	if callCfg.MaxTokens != nil {
		maxTokens = *callCfg.MaxTokens
	}

	body := AnthropicRequest{
		Model:       callCfg.Model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: callCfg.Temperature,
	}

	// Add system message if provided
	if callCfg.SystemMsg != "" {
		body.System = callCfg.SystemMsg
	}

	resp := AnthropicResponse{}
	err := callHTTPAPI(ctx, callCfg.BaseURL, func(req *http.Request) {
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("x-api-key", c.apiKey)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("api call failed: %w", err)
	}

	// Extract text from response
	if len(resp.Content) == 0 {
		return nil, fmt.Errorf("no content in Anthropic response")
	}

	// Combine all text content
	var text string
	for _, content := range resp.Content {
		if content.Type == "text" {
			text += content.Text
		}
	}

	return &Response{
		Text: text,
		Metadata: map[string]any{
			"stop_reason":   resp.StopReason,
			"input_tokens":  resp.Usage.InputTokens,
			"output_tokens": resp.Usage.OutputTokens,
		},
	}, nil
}
