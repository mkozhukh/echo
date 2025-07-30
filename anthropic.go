package echo

import (
	"context"
	"encoding/json"
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
	Stream      bool               `json:"stream,omitempty"`
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

// Anthropic streaming response structures
type AnthropicStreamEvent struct {
	Type string `json:"type"`
}

type AnthropicMessageStart struct {
	Type    string `json:"type"`
	Message struct {
		ID           string  `json:"id"`
		Type         string  `json:"type"`
		Role         string  `json:"role"`
		Content      []any   `json:"content"`
		Model        string  `json:"model"`
		StopReason   *string `json:"stop_reason"`
		StopSequence *string `json:"stop_sequence"`
		Usage        struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	} `json:"message"`
}

type AnthropicContentBlockStart struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content_block"`
}

type AnthropicContentBlockDelta struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

type AnthropicContentBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type AnthropicMessageDelta struct {
	Type  string `json:"type"`
	Delta struct {
		StopReason   *string `json:"stop_reason"`
		StopSequence *string `json:"stop_sequence"`
	} `json:"delta"`
	Usage *struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
}

type AnthropicMessageStop struct {
	Type string `json:"type"`
}

type AnthropicPing struct {
	Type string `json:"type"`
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

// prepareRequest builds the Anthropic request with the given configuration
func (c *AnthropicClient) prepareRequest(prompt string, streaming bool, opts ...CallOption) (AnthropicRequest, CallConfig) {
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
		Stream:      streaming,
	}

	// Add system message if provided
	if callCfg.SystemMsg != "" {
		body.System = callCfg.SystemMsg
	}

	return body, callCfg
}

func (c *AnthropicClient) Call(ctx context.Context, prompt string, opts ...CallOption) (*Response, error) {
	body, callCfg := c.prepareRequest(prompt, false, opts...)

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

func (c *AnthropicClient) StreamCall(ctx context.Context, prompt string, opts ...CallOption) (*StreamResponse, error) {
	body, callCfg := c.prepareRequest(prompt, true, opts...)

	// Get streaming response
	respBody, err := streamHTTPAPI(ctx, callCfg.BaseURL, func(req *http.Request) {
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("x-api-key", c.apiKey)
	}, body)
	if err != nil {
		return nil, fmt.Errorf("Anthropic streaming API call failed: %w", err)
	}

	// Create channel for streaming
	ch := make(chan StreamChunk)

	// Start goroutine to process stream
	go func() {
		defer close(ch)

		var totalInputTokens, totalOutputTokens int

		err := parseSSEStream(respBody, func(msg SSEMessage) error {
			return c.processAnthropicSSEMessage(msg, ch, &totalInputTokens, &totalOutputTokens)
		})

		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("SSE stream error: %w", err)}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}

// processAnthropicSSEMessage processes individual Anthropic SSE messages
func (c *AnthropicClient) processAnthropicSSEMessage(msg SSEMessage, ch chan StreamChunk, totalInputTokens, totalOutputTokens *int) error {
	if len(msg.Data) == 0 {
		return nil
	}

	// Handle different event types
	switch msg.Event {
	case "message_start":
		var messageStart AnthropicMessageStart
		if err := json.Unmarshal(msg.Data, &messageStart); err != nil {
			return fmt.Errorf("json parse error for message_start: %w", err)
		}
		// Store initial token counts
		*totalInputTokens = messageStart.Message.Usage.InputTokens
		*totalOutputTokens = messageStart.Message.Usage.OutputTokens

	case "content_block_start":
		// Content block started, no action needed

	case "content_block_delta":
		var contentDelta AnthropicContentBlockDelta
		if err := json.Unmarshal(msg.Data, &contentDelta); err != nil {
			return fmt.Errorf("json parse error for content_block_delta: %w", err)
		}
		// Send the text delta
		if contentDelta.Delta.Type == "text_delta" && contentDelta.Delta.Text != "" {
			ch <- StreamChunk{
				Data: contentDelta.Delta.Text,
			}
		}

	case "content_block_stop":
		// Content block finished, no action needed

	case "message_delta":
		var messageDelta AnthropicMessageDelta
		if err := json.Unmarshal(msg.Data, &messageDelta); err != nil {
			return fmt.Errorf("json parse error for message_delta: %w", err)
		}
		// Update output token count if provided
		if messageDelta.Usage != nil {
			*totalOutputTokens = messageDelta.Usage.OutputTokens
		}

	case "message_stop":
		// Send final metadata
		meta := Metadata{
			"input_tokens":  *totalInputTokens,
			"output_tokens": *totalOutputTokens,
		}
		ch <- StreamChunk{
			Meta: &meta,
		}

	case "ping":
		// Ping event, ignore

	default:
		// Try to parse as generic event to handle cases without event type
		var genericEvent AnthropicStreamEvent
		if err := json.Unmarshal(msg.Data, &genericEvent); err != nil {
			return nil // Skip unparseable events (not an error)
		}

		// Handle based on type field in data
		switch genericEvent.Type {
		case "content_block_delta":
			var contentDelta AnthropicContentBlockDelta
			if err := json.Unmarshal(msg.Data, &contentDelta); err == nil {
				if contentDelta.Delta.Type == "text_delta" && contentDelta.Delta.Text != "" {
					ch <- StreamChunk{
						Data: contentDelta.Delta.Text,
					}
				}
			}
		case "message_delta":
			var messageDelta AnthropicMessageDelta
			if err := json.Unmarshal(msg.Data, &messageDelta); err == nil {
				if messageDelta.Usage != nil {
					*totalOutputTokens = messageDelta.Usage.OutputTokens
				}
			}
		case "message_stop":
			meta := Metadata{
				"input_tokens":  *totalInputTokens,
				"output_tokens": *totalOutputTokens,
			}
			ch <- StreamChunk{
				Meta: &meta,
			}
		}
	}

	return nil
}
