package echo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// MockProvider is a stateless provider for mock testing
type MockProvider struct{}

func (p *MockProvider) getMessages(messages []Message, cfg CallConfig) string {
	if len(messages) > 0 && messages[0].Role == "system" {
		if cfg.SystemMsg != "" {
			messages[0].Content = cfg.SystemMsg
		}
	} else {
		if cfg.SystemMsg != "" {
			messages = append([]Message{{Role: "system", Content: cfg.SystemMsg}}, messages...)
		}
	}

	// Combine all message content
	var combinedContent strings.Builder
	for i, msg := range messages {
		if i > 0 {
			combinedContent.WriteString("\n")
		}
		combinedContent.WriteString(fmt.Sprintf("[%s]: %s", msg.Role, msg.Content))
	}

	return combinedContent.String()
}

// call implements the provider interface for mock testing
func (p *MockProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return nil, fmt.Errorf("invalid message chain: %w", err)
	}

	return &Response{
		Text: p.getMessages(messages, cfg),
		Metadata: Metadata{
			"mock":          true,
			"message_count": len(messages),
		},
	}, nil
}

// streamCall implements the provider interface for mock streaming
func (p *MockProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return nil, fmt.Errorf("invalid message chain: %w", err)
	}

	// Create channel for streaming
	ch := make(chan StreamChunk)

	// Start goroutine to simulate streaming
	go func() {
		defer close(ch)

		// Send metadata in first chunk
		ch <- StreamChunk{
			Meta: &Metadata{
				"mock":          true,
				"message_count": len(messages),
			},
		}

		// Simulate streaming by sending the combined content in chunks
		content := p.getMessages(messages, cfg)
		chunkSize := 10 // Send 10 characters at a time for simulation

		for i := 0; i < len(content); i += chunkSize {
			end := i + chunkSize
			if end > len(content) {
				end = len(content)
			}

			ch <- StreamChunk{
				Data: content[i:end],
			}
		}

		// Send completion signal
		ch <- StreamChunk{
			Error: nil, // nil error indicates completion
		}
	}()

	return &StreamResponse{
		Stream: ch,
	}, nil
}

// getEmbeddings implements the provider interface for mock embeddings
func (p *MockProvider) getEmbeddings(ctx context.Context, text string, cfg CallConfig) (*EmbeddingResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

// reRank implements the provider interface for mock reranking
func (p *MockProvider) reRank(ctx context.Context, query string, documents []string, cfg CallConfig) (*RerankResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

// parseCompletionRequest parses an HTTP request into a CompletionRequest
func (p *MockProvider) parseCompletionRequest(req *http.Request) (*CompletionRequest, error) {
	var completionReq CompletionRequest
	if err := json.NewDecoder(req.Body).Decode(&completionReq); err != nil {
		return nil, fmt.Errorf("failed to parse mock completion request: %w", err)
	}
	return &completionReq, nil
}

// parseEmbeddingRequest parses an HTTP request into an EmbeddingRequest
func (p *MockProvider) parseEmbeddingRequest(req *http.Request) (*EmbeddingRequest, error) {
	return nil, fmt.Errorf("not implemented")
}

// parseRerankRequest parses an HTTP request into a RerankRequest
func (p *MockProvider) parseRerankRequest(req *http.Request) (*RerankRequest, error) {
	return nil, fmt.Errorf("not implemented")
}

// buildCompletionRequest builds and executes a completion request, returning a unified response
func (p *MockProvider) buildCompletionRequest(ctx context.Context, req *CompletionRequest, cfg CallConfig) (*CompletionResponse, error) {
	// Create mock response with combined message content
	var combinedContent strings.Builder
	for i, msg := range req.Messages {
		if i > 0 {
			combinedContent.WriteString("\n")
		}
		combinedContent.WriteString(fmt.Sprintf("[%s]: %s", msg.Role, msg.Content))
	}

	// Create unified completion response
	completionResp := &CompletionResponse{
		ID:      "mock-completion-id",
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

	completionResp.Choices[0].Index = 0
	completionResp.Choices[0].Message.Role = "assistant"
	completionResp.Choices[0].Message.Content = combinedContent.String()
	completionResp.Choices[0].FinishReason = "stop"

	// Add mock usage
	completionResp.Usage = &struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}{
		PromptTokens:     len(req.Messages) * 10,
		CompletionTokens: 20,
		TotalTokens:      len(req.Messages)*10 + 20,
	}

	return completionResp, nil
}

// buildEmbeddingRequest builds and executes an embedding request, returning a unified response
func (p *MockProvider) buildEmbeddingRequest(ctx context.Context, req *EmbeddingRequest, cfg CallConfig) (*UnifiedEmbeddingResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

// buildRerankRequest builds and executes a reranking request, returning a unified response
func (p *MockProvider) buildRerankRequest(ctx context.Context, req *RerankRequest, cfg CallConfig) (*UnifiedRerankResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

// writeCompletionResponse writes a CompletionResponse as JSON to the HTTP response writer
func (p *MockProvider) writeCompletionResponse(w http.ResponseWriter, resp *CompletionResponse) error {
	w.Header().Set("Content-Type", "plain/text")
	_, err := w.Write([]byte(resp.Choices[0].Message.Content))
	return err
}

// writeEmbeddingResponse writes a UnifiedEmbeddingResponse as JSON to the HTTP response writer
func (p *MockProvider) writeEmbeddingResponse(w http.ResponseWriter, resp *UnifiedEmbeddingResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

// writeRerankResponse writes a UnifiedRerankResponse as JSON to the HTTP response writer
func (p *MockProvider) writeRerankResponse(w http.ResponseWriter, resp *UnifiedRerankResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}
