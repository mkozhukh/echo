package echo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicRequest struct {
	Model        string                 `json:"model"`
	Messages     []anthropicMessage     `json:"messages"`
	MaxTokens    int                    `json:"max_tokens"`
	Temperature  *float32               `json:"temperature,omitempty"`
	System       string                 `json:"system,omitempty"`
	Stream       bool                   `json:"stream,omitempty"`
	OutputFormat *anthropicOutputFormat `json:"output_format,omitempty"`
	OutputConfig *anthropicOutputConfig `json:"output_config,omitempty"`
}

// anthropicOutputFormat specifies the output format for structured output
type anthropicOutputFormat struct {
	Type   string `json:"type"`   // "json_schema"
	Schema any    `json:"schema"` // JSON Schema
}

// anthropicOutputConfig specifies output configuration for reasoning effort
type anthropicOutputConfig struct {
	Effort string `json:"effort"` // "low", "medium", "high"
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicResponse struct {
	Error   *anthropicError `json:"error,omitempty"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      anthropicUsage `json:"usage"`
}

// anthropicUsage is the usage block of Anthropic responses.
// input_tokens excludes cached tokens, which are reported separately.
type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

func (u anthropicUsage) toUsage() *Usage {
	input := u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
	return &Usage{
		InputTokens:       input,
		CachedInputTokens: u.CacheReadInputTokens,
		CacheWriteTokens:  u.CacheCreationInputTokens,
		OutputTokens:      u.OutputTokens,
		TotalTokens:       input + u.OutputTokens,
	}
}

// merge applies a message_delta usage update: output tokens are cumulative,
// other fields are updated only when present.
func (u *anthropicUsage) merge(d *anthropicUsage) {
	if d == nil {
		return
	}
	u.OutputTokens = d.OutputTokens
	if d.InputTokens > 0 {
		u.InputTokens = d.InputTokens
	}
	if d.CacheCreationInputTokens > 0 {
		u.CacheCreationInputTokens = d.CacheCreationInputTokens
	}
	if d.CacheReadInputTokens > 0 {
		u.CacheReadInputTokens = d.CacheReadInputTokens
	}
}

// Anthropic streaming response structures
type anthropicStreamEvent struct {
	Type string `json:"type"`
}

type anthropicMessageStart struct {
	Type    string `json:"type"`
	Message struct {
		ID           string         `json:"id"`
		Type         string         `json:"type"`
		Role         string         `json:"role"`
		Content      []any          `json:"content"`
		Model        string         `json:"model"`
		StopReason   *string        `json:"stop_reason"`
		StopSequence *string        `json:"stop_sequence"`
		Usage        anthropicUsage `json:"usage"`
	} `json:"message"`
}

type anthropicContentBlockDelta struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

type anthropicMessageDelta struct {
	Type  string `json:"type"`
	Delta struct {
		StopReason   *string `json:"stop_reason"`
		StopSequence *string `json:"stop_sequence"`
	} `json:"delta"`
	Usage *anthropicUsage `json:"usage,omitempty"`
}

// AnthropicProvider is a stateless provider for Anthropic API
type AnthropicProvider struct {
	Key string
}

// NewAnthropicClient creates a new Anthropic client (deprecated, kept for compatibility)
func NewAnthropicClient(apiKey, model string, opts ...CallOption) Client {
	if model != "" {
		if !strings.HasPrefix(model, "anthropic/") {
			model = "anthropic/" + model
		}
		opts = append([]CallOption{WithModel(model)}, opts...)
	}
	client, _ := NewClient(opts...)
	client.SetProvider("anthropic", &AnthropicProvider{Key: apiKey})
	return client
}

// prepareAnthropicRequest builds the Anthropic request with the given configuration
func prepareAnthropicRequest(messages []Message, streaming bool, cfg CallConfig) (anthropicRequest, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return anthropicRequest{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Convert messages to Anthropic format
	anthropicMessages := []anthropicMessage{}
	var systemMsg string

	for _, msg := range messages {
		switch msg.Role {
		case System:
			systemMsg = msg.Content
		case User:
			anthropicMessages = append(anthropicMessages, anthropicMessage{
				Role:    "user",
				Content: msg.Content,
			})
		case Agent:
			anthropicMessages = append(anthropicMessages, anthropicMessage{
				Role:    "assistant",
				Content: msg.Content,
			})
		}
	}

	// Anthropic requires max_tokens to be set
	maxTokens := 4096
	if cfg.MaxTokens != nil {
		maxTokens = *cfg.MaxTokens
	}

	body := anthropicRequest{
		Model:       cfg.Model,
		Messages:    anthropicMessages,
		MaxTokens:   maxTokens,
		Temperature: cfg.Temperature,
		Stream:      streaming,
	}

	// Handle system message - WithSystemMessage overrides message chain system
	if cfg.SystemMsg != "" {
		body.System = cfg.SystemMsg
	} else if systemMsg != "" {
		body.System = systemMsg
	}

	// Handle structured output via native output_format API
	if cfg.StructuredOutput != nil {
		body.OutputFormat = &anthropicOutputFormat{
			Type:   "json_schema",
			Schema: cfg.StructuredOutput.Schema,
		}
	}

	// Handle reasoning effort via output_config
	if cfg.ReasoningEffort != "" {
		body.OutputConfig = &anthropicOutputConfig{
			Effort: cfg.ReasoningEffort,
		}
	}

	return body, nil
}

// call implements the provider interface for Anthropic
func (p *AnthropicProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
	body, err := prepareAnthropicRequest(messages, false, cfg)
	if err != nil {
		return nil, err
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1/messages"
	}

	resp := anthropicResponse{}
	err = callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("x-api-key", p.Key)
		// Add beta headers for features that require them
		var betaFeatures []string
		if cfg.StructuredOutput != nil {
			betaFeatures = append(betaFeatures, "structured-outputs-2025-11-13")
		}
		if cfg.ReasoningEffort != "" {
			betaFeatures = append(betaFeatures, "effort-2025-11-24")
		}
		if len(betaFeatures) > 0 {
			req.Header.Set("anthropic-beta", strings.Join(betaFeatures, ","))
		}
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("api call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("Anthropic API error: %s", resp.Error.Message)
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
		Text:  text,
		Usage: resp.Usage.toUsage(),
		Metadata: Metadata{
			"stop_reason": resp.StopReason,
		},
	}, nil
}

// streamCall implements the provider interface for Anthropic streaming
func (p *AnthropicProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	body, err := prepareAnthropicRequest(messages, true, cfg)
	if err != nil {
		return nil, err
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1/messages"
	}

	// Get streaming response
	respBody, err := streamHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("x-api-key", p.Key)
		// Add beta headers for features that require them
		var betaFeatures []string
		if cfg.StructuredOutput != nil {
			betaFeatures = append(betaFeatures, "structured-outputs-2025-11-13")
		}
		if cfg.ReasoningEffort != "" {
			betaFeatures = append(betaFeatures, "effort-2025-11-24")
		}
		if len(betaFeatures) > 0 {
			req.Header.Set("anthropic-beta", strings.Join(betaFeatures, ","))
		}
	}, body)
	if err != nil {
		return nil, fmt.Errorf("Anthropic streaming API call failed: %w", err)
	}

	// Create channel for streaming
	ch := make(chan StreamChunk)

	// Start goroutine to process stream
	go func() {
		defer close(ch)

		var usage anthropicUsage

		err := parseSSEStream(respBody, func(msg sseMessage) error {
			return processAnthropicSSEMessage(msg, ch, &usage)
		})

		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("SSE stream error: %w", err)}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}

// processAnthropicSSEMessage processes individual Anthropic SSE messages
func processAnthropicSSEMessage(msg sseMessage, ch chan StreamChunk, usage *anthropicUsage) error {
	if len(msg.Data) == 0 {
		return nil
	}

	// Handle different event types
	switch msg.Event {
	case "message_start":
		var messageStart anthropicMessageStart
		if err := json.Unmarshal(msg.Data, &messageStart); err != nil {
			return fmt.Errorf("json parse error for message_start: %w", err)
		}
		// Store initial token counts
		*usage = messageStart.Message.Usage

	case "content_block_start":
		// Content block started, no action needed

	case "content_block_delta":
		var contentDelta anthropicContentBlockDelta
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
		var messageDelta anthropicMessageDelta
		if err := json.Unmarshal(msg.Data, &messageDelta); err != nil {
			return fmt.Errorf("json parse error for message_delta: %w", err)
		}
		// Update token counts if provided
		usage.merge(messageDelta.Usage)

	case "message_stop":
		// Send final usage
		ch <- StreamChunk{
			Usage: usage.toUsage(),
		}

	case "ping":
		// Ping event, ignore

	default:
		// Try to parse as generic event to handle cases without event type
		var genericEvent anthropicStreamEvent
		if err := json.Unmarshal(msg.Data, &genericEvent); err != nil {
			return nil // Skip unparseable events (not an error)
		}

		// Handle based on type field in data
		switch genericEvent.Type {
		case "content_block_delta":
			var contentDelta anthropicContentBlockDelta
			if err := json.Unmarshal(msg.Data, &contentDelta); err == nil {
				if contentDelta.Delta.Type == "text_delta" && contentDelta.Delta.Text != "" {
					ch <- StreamChunk{
						Data: contentDelta.Delta.Text,
					}
				}
			}
		case "message_delta":
			var messageDelta anthropicMessageDelta
			if err := json.Unmarshal(msg.Data, &messageDelta); err == nil {
				usage.merge(messageDelta.Usage)
			}
		case "message_stop":
			ch <- StreamChunk{
				Usage: usage.toUsage(),
			}
		}
	}

	return nil
}
