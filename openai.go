package echo

import (
	"context"
	"fmt"
	"net/http"
)

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIRequest struct {
	Model       string          `json:"model"`
	Temperature *float64        `json:"temperature,omitempty"`
	MaxTokens   *int            `json:"max_tokens,omitempty"`
	Messages    []OpenAIMessage `json:"messages"`
}

type OpenAIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
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

func NewGPT41(apiKey string, opts ...CallOption) *OpenAIClient {
	return NewOpenAIClient(apiKey, "gpt-4.1", opts...)
}

func NewGPT41Mini(apiKey string, opts ...CallOption) *OpenAIClient {
	return NewOpenAIClient(apiKey, "gpt-4.1-mini", opts...)
}

func NewGPT41Nano(apiKey string, opts ...CallOption) *OpenAIClient {
	return NewOpenAIClient(apiKey, "gpt-4.1-nano", opts...)
}

func (c *OpenAIClient) Call(ctx context.Context, prompt string, opts ...CallOption) (*Response, error) {
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

	body := OpenAIRequest{
		Model:       callCfg.Model,
		Temperature: callCfg.Temperature,
		MaxTokens:   callCfg.MaxTokens,
		Messages:    messages,
	}

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

	return &Response{
		Text: resp.Choices[0].Message.Content,
	}, nil
}
