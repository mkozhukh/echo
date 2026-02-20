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

// XAIRequest represents a request to the xAI chat completions API
type XAIRequest struct {
	Model         string          `json:"model"`
	Temperature   *float32        `json:"temperature,omitempty"`
	MaxTokens     *int            `json:"max_completion_tokens,omitempty"`
	Messages      []OpenAIMessage `json:"messages"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
	ResponseFormat  *OpenAIResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort string                `json:"reasoning_effort,omitempty"`
	Store           *bool                 `json:"store,omitempty"` // xAI-specific: set to false to disable server-side storage
}

// XAIError represents an error from the xAI API
type XAIError struct {
	Message string `json:"message"`
	Code    int32  `json:"code"`
}

// XAIResponse represents a response from the xAI chat completions API
type XAIResponse struct {
	Error   *XAIError `json:"error,omitempty"`
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

// XAIStreamResponse represents a streaming response chunk from xAI
type XAIStreamResponse struct {
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

// XAIProvider is a stateless provider for xAI (Grok) API
type XAIProvider struct {
	Key string
}

// NewXAIClient creates a new xAI client
func NewXAIClient(apiKey, model string, opts ...CallOption) Client {
	client, _ := NewClient(opts...)
	client.SetProvider("xai", &XAIProvider{Key: apiKey})
	return client
}

// prepareXAIRequest builds the xAI request with the given configuration
func prepareXAIRequest(messages []Message, streaming bool, cfg CallConfig) (XAIRequest, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return XAIRequest{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Convert messages to OpenAI format (xAI is OpenAI-compatible)
	xaiMessages := []OpenAIMessage{}
	systemMessageProcessed := false

	for _, msg := range messages {
		switch msg.Role {
		case System:
			// Skip system message here if WithSystemMessage is set
			if cfg.SystemMsg == "" {
				xaiMessages = append(xaiMessages, OpenAIMessage{
					Role:    "system",
					Content: msg.Content,
				})
			}
			systemMessageProcessed = true
		case User:
			xaiMessages = append(xaiMessages, OpenAIMessage{
				Role:    "user",
				Content: msg.Content,
			})
		case Agent:
			xaiMessages = append(xaiMessages, OpenAIMessage{
				Role:    "assistant",
				Content: msg.Content,
			})
		}
	}

	// Handle WithSystemMessage option
	if cfg.SystemMsg != "" {
		// Insert system message at the beginning
		systemMsg := OpenAIMessage{
			Role:    "system",
			Content: cfg.SystemMsg,
		}
		if systemMessageProcessed {
			// Replace the first message (which should be system)
			xaiMessages = append([]OpenAIMessage{systemMsg}, xaiMessages[1:]...)
		} else {
			// Add system message at the beginning
			xaiMessages = append([]OpenAIMessage{systemMsg}, xaiMessages...)
		}
	}

	req := XAIRequest{
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
		req.ResponseFormat = &OpenAIResponseFormat{
			Type: "json_schema",
			JSONSchema: &OpenAIJSONSchemaConfig{
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

	resp := XAIResponse{}
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
			var streamResp XAIStreamResponse
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

// getEmbeddings implements the provider interface for xAI
// Note: xAI embedding API support TBD
func (p *XAIProvider) getEmbeddings(ctx context.Context, text string, cfg CallConfig) (*EmbeddingResponse, error) {
	return nil, fmt.Errorf("xAI does not currently support embeddings API")
}

// reRank implements the provider interface for xAI
// Note: xAI does not support reranking API
func (p *XAIProvider) reRank(ctx context.Context, query string, documents []string, cfg CallConfig) (*RerankResponse, error) {
	return nil, fmt.Errorf("xAI does not support reranking API")
}

// parseCompletionRequest parses an HTTP request into a CompletionRequest
// For xAI, we use OpenAI format as the common format
func (p *XAIProvider) parseCompletionRequest(req *http.Request) (*CompletionRequest, error) {
	var completionReq CompletionRequest
	if err := json.NewDecoder(req.Body).Decode(&completionReq); err != nil {
		return nil, fmt.Errorf("failed to parse completion request: %w", err)
	}
	return &completionReq, nil
}

// parseEmbeddingRequest parses an HTTP request into an EmbeddingRequest
// xAI does not support embeddings, so this returns an error
func (p *XAIProvider) parseEmbeddingRequest(req *http.Request) (*EmbeddingRequest, error) {
	return nil, fmt.Errorf("xAI does not currently support embeddings API")
}

// parseRerankRequest parses an HTTP request into a RerankRequest
// xAI does not support reranking, so this returns an error
func (p *XAIProvider) parseRerankRequest(req *http.Request) (*RerankRequest, error) {
	return nil, fmt.Errorf("xAI does not support reranking API")
}

// buildCompletionRequest builds and executes a completion request, returning a unified response
func (p *XAIProvider) buildCompletionRequest(ctx context.Context, req *CompletionRequest, cfg CallConfig) (*CompletionResponse, error) {
	// Convert CompletionRequest to XAIRequest
	xaiReq := XAIRequest{
		Model:         req.Model,
		Temperature:   req.Temperature,
		MaxTokens:     req.MaxTokens,
		Messages:      req.Messages,
		Stream:        req.Stream,
		StreamOptions: req.StreamOptions,
	}

	// Handle store parameter - default to false for privacy
	if cfg.StoreData != nil {
		xaiReq.Store = cfg.StoreData
	} else {
		// Default to false (disable server-side storage)
		store := false
		xaiReq.Store = &store
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.x.ai/v1/chat/completions"
	}

	// Make the API call
	var xaiResp XAIResponse
	err := callHTTPAPI(ctx, baseURL, func(httpReq *http.Request) {
		httpReq.Header.Set("Authorization", "Bearer "+p.Key)
	}, xaiReq, &xaiResp)
	if err != nil {
		return nil, fmt.Errorf("xAI API call failed: %w", err)
	}

	// Check for errors in the response
	if xaiResp.Error != nil {
		return nil, fmt.Errorf("xAI API error: %s", xaiResp.Error.Message)
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
		}, len(xaiResp.Choices)),
	}

	// Copy choices
	for i, choice := range xaiResp.Choices {
		completionResp.Choices[i].Index = i
		completionResp.Choices[i].Message.Role = "assistant"
		completionResp.Choices[i].Message.Content = choice.Message.Content
		completionResp.Choices[i].FinishReason = "stop"
	}

	// Copy usage if available
	if xaiResp.Usage != nil {
		completionResp.Usage = &struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     xaiResp.Usage.PromptTokens,
			CompletionTokens: xaiResp.Usage.CompletionTokens,
			TotalTokens:      xaiResp.Usage.TotalTokens,
		}
	}

	return completionResp, nil
}

// buildEmbeddingRequest builds and executes an embedding request, returning a unified response
// xAI does not support embeddings, so this returns an error
func (p *XAIProvider) buildEmbeddingRequest(ctx context.Context, req *EmbeddingRequest, cfg CallConfig) (*UnifiedEmbeddingResponse, error) {
	return nil, fmt.Errorf("xAI does not currently support embeddings API")
}

// buildRerankRequest builds and executes a reranking request, returning a unified response
// xAI does not support reranking, so this returns an error
func (p *XAIProvider) buildRerankRequest(ctx context.Context, req *RerankRequest, cfg CallConfig) (*UnifiedRerankResponse, error) {
	return nil, fmt.Errorf("xAI does not support reranking API")
}

// writeCompletionResponse writes a CompletionResponse as JSON to the HTTP response writer
func (p *XAIProvider) writeCompletionResponse(w http.ResponseWriter, resp *CompletionResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

// writeEmbeddingResponse writes a UnifiedEmbeddingResponse as JSON to the HTTP response writer
// xAI does not support embeddings, so this returns an error
func (p *XAIProvider) writeEmbeddingResponse(w http.ResponseWriter, resp *UnifiedEmbeddingResponse) error {
	return fmt.Errorf("xAI does not currently support embeddings API")
}

// writeRerankResponse writes a UnifiedRerankResponse as JSON to the HTTP response writer
// xAI does not support reranking, so this returns an error
func (p *XAIProvider) writeRerankResponse(w http.ResponseWriter, resp *UnifiedRerankResponse) error {
	return fmt.Errorf("xAI does not support reranking API")
}
