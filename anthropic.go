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

type AnthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type AnthropicResponse struct {
	Error   *AnthropicError `json:"error,omitempty"`
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

// AnthropicProvider is a stateless provider for Anthropic API
type AnthropicProvider struct {
	Key string
}

// NewAnthropicClient creates a new Anthropic client (deprecated, kept for compatibility)
func NewAnthropicClient(apiKey, model string, opts ...CallOption) Client {
	client, _ := NewClient(opts...)
	client.SetProvider("anthropic", &AnthropicProvider{Key: apiKey})
	return client
}

// prepareAnthropicRequest builds the Anthropic request with the given configuration
func prepareAnthropicRequest(messages []Message, streaming bool, cfg CallConfig) (AnthropicRequest, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return AnthropicRequest{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Convert messages to Anthropic format
	anthropicMessages := []AnthropicMessage{}
	var systemMsg string

	for _, msg := range messages {
		switch msg.Role {
		case System:
			systemMsg = msg.Content
		case User:
			anthropicMessages = append(anthropicMessages, AnthropicMessage{
				Role:    "user",
				Content: msg.Content,
			})
		case Agent:
			anthropicMessages = append(anthropicMessages, AnthropicMessage{
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

	body := AnthropicRequest{
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

	resp := AnthropicResponse{}
	err = callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("x-api-key", p.Key)
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
		Text: text,
		Metadata: map[string]any{
			"stop_reason":   resp.StopReason,
			"input_tokens":  resp.Usage.InputTokens,
			"output_tokens": resp.Usage.OutputTokens,
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
			return processAnthropicSSEMessage(msg, ch, &totalInputTokens, &totalOutputTokens)
		})

		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("SSE stream error: %w", err)}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}

// processAnthropicSSEMessage processes individual Anthropic SSE messages
func processAnthropicSSEMessage(msg SSEMessage, ch chan StreamChunk, totalInputTokens, totalOutputTokens *int) error {
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

// getEmbeddings implements the provider interface for Anthropic
// Note: Anthropic does not currently support embeddings API
func (p *AnthropicProvider) getEmbeddings(ctx context.Context, text string, cfg CallConfig) (*EmbeddingResponse, error) {
	return nil, fmt.Errorf("Anthropic does not support embeddings API")
}

// reRank implements the provider interface for Anthropic
// Note: Anthropic does not currently support reranking API
func (p *AnthropicProvider) reRank(ctx context.Context, query string, documents []string, cfg CallConfig) (*RerankResponse, error) {
	return nil, fmt.Errorf("Anthropic does not support reranking API")
}

// parseCompletionRequest parses an HTTP request into a CompletionRequest
// Converts from Anthropic format to OpenAI-compatible format
func (p *AnthropicProvider) parseCompletionRequest(req *http.Request) (*CompletionRequest, error) {
	var anthropicReq AnthropicRequest
	if err := json.NewDecoder(req.Body).Decode(&anthropicReq); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic completion request: %w", err)
	}

	// Convert Anthropic messages to OpenAI format
	messages := make([]OpenAIMessage, 0, len(anthropicReq.Messages)+1)

	// Add system message as first message if present
	if anthropicReq.System != "" {
		messages = append(messages, OpenAIMessage{
			Role:    "system",
			Content: anthropicReq.System,
		})
	}

	// Convert user/assistant messages
	for _, msg := range anthropicReq.Messages {
		messages = append(messages, OpenAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Convert max_tokens to OpenAI format
	var maxTokens *int
	if anthropicReq.MaxTokens > 0 {
		maxTokens = &anthropicReq.MaxTokens
	}

	completionReq := &CompletionRequest{
		Model:       anthropicReq.Model,
		Temperature: anthropicReq.Temperature,
		MaxTokens:   maxTokens,
		Messages:    messages,
		Stream:      anthropicReq.Stream,
	}

	if anthropicReq.Stream {
		completionReq.StreamOptions = &struct {
			IncludeUsage bool `json:"include_usage"`
		}{
			IncludeUsage: true,
		}
	}

	return completionReq, nil
}

// parseEmbeddingRequest parses an HTTP request into an EmbeddingRequest
// Anthropic does not support embeddings, so this returns an error
func (p *AnthropicProvider) parseEmbeddingRequest(req *http.Request) (*EmbeddingRequest, error) {
	return nil, fmt.Errorf("Anthropic does not support embeddings API")
}

// parseRerankRequest parses an HTTP request into a RerankRequest
// Anthropic does not support reranking, so this returns an error
func (p *AnthropicProvider) parseRerankRequest(req *http.Request) (*RerankRequest, error) {
	return nil, fmt.Errorf("Anthropic does not support reranking API")
}

// buildCompletionRequest builds and executes a completion request, returning a unified response
func (p *AnthropicProvider) buildCompletionRequest(ctx context.Context, req *CompletionRequest, cfg CallConfig) (*CompletionResponse, error) {
	// Convert CompletionRequest to AnthropicRequest
	anthropicReq := AnthropicRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   4096, // Default
		Stream:      req.Stream,
	}

	// Override max tokens if provided
	if req.MaxTokens != nil {
		anthropicReq.MaxTokens = *req.MaxTokens
	}

	// Convert messages
	var systemMsg string
	anthropicReq.Messages = make([]AnthropicMessage, 0, len(req.Messages))
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			systemMsg = msg.Content
		} else {
			anthropicReq.Messages = append(anthropicReq.Messages, AnthropicMessage{
				Role:    msg.Role, // "user" or "assistant"
				Content: msg.Content,
			})
		}
	}
	if systemMsg != "" {
		anthropicReq.System = systemMsg
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1/messages"
	}

	// Make the API call
	var anthropicResp AnthropicResponse
	err := callHTTPAPI(ctx, baseURL, func(httpReq *http.Request) {
		httpReq.Header.Set("anthropic-version", "2023-06-01")
		httpReq.Header.Set("x-api-key", p.Key)
	}, anthropicReq, &anthropicResp)
	if err != nil {
		return nil, fmt.Errorf("Anthropic API call failed: %w", err)
	}

	// Check for errors in the response
	if anthropicResp.Error != nil {
		return nil, fmt.Errorf("Anthropic API error: %s", anthropicResp.Error.Message)
	}

	// Convert to unified CompletionResponse
	completionResp := &CompletionResponse{
		ID:      "",
		Object:  "chat.completion",
		Created: 0,
		Model:   req.Model,
		Choices: make([]struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason,omitempty"`
		}, 1),
	}

	// Combine all text content
	var text string
	for _, content := range anthropicResp.Content {
		if content.Type == "text" {
			text += content.Text
		}
	}

	completionResp.Choices[0].Index = 0
	completionResp.Choices[0].Message.Role = "assistant"
	completionResp.Choices[0].Message.Content = text
	completionResp.Choices[0].FinishReason = anthropicResp.StopReason

	// Add usage information
	completionResp.Usage = &struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}{
		PromptTokens:     anthropicResp.Usage.InputTokens,
		CompletionTokens: anthropicResp.Usage.OutputTokens,
		TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
	}

	return completionResp, nil
}

// buildEmbeddingRequest builds and executes an embedding request, returning a unified response
// Anthropic does not support embeddings, so this returns an error
func (p *AnthropicProvider) buildEmbeddingRequest(ctx context.Context, req *EmbeddingRequest, cfg CallConfig) (*UnifiedEmbeddingResponse, error) {
	return nil, fmt.Errorf("Anthropic does not support embeddings API")
}

// buildRerankRequest builds and executes a reranking request, returning a unified response
// Anthropic does not support reranking, so this returns an error
func (p *AnthropicProvider) buildRerankRequest(ctx context.Context, req *RerankRequest, cfg CallConfig) (*UnifiedRerankResponse, error) {
	return nil, fmt.Errorf("Anthropic does not support reranking API")
}

// writeCompletionResponse writes a CompletionResponse as JSON to the HTTP response writer
func (p *AnthropicProvider) writeCompletionResponse(w http.ResponseWriter, resp *CompletionResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

// writeEmbeddingResponse writes a UnifiedEmbeddingResponse as JSON to the HTTP response writer
// Anthropic does not support embeddings, so this returns an error
func (p *AnthropicProvider) writeEmbeddingResponse(w http.ResponseWriter, resp *UnifiedEmbeddingResponse) error {
	return fmt.Errorf("Anthropic does not support embeddings API")
}

// writeRerankResponse writes a UnifiedRerankResponse as JSON to the HTTP response writer
// Anthropic does not support reranking, so this returns an error
func (p *AnthropicProvider) writeRerankResponse(w http.ResponseWriter, resp *UnifiedRerankResponse) error {
	return fmt.Errorf("Anthropic does not support reranking API")
}
