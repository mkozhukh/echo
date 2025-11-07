package echo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// mockProvider is a stateless provider for mock testing
type mockProvider struct{}

func (p *mockProvider) getMessages(messages []Message, cfg CallConfig) string {
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
func (p *mockProvider) call(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*Response, error) {
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
func (p *mockProvider) streamCall(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*StreamResponse, error) {
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
func (p *mockProvider) getEmbeddings(ctx context.Context, apiKey string, text string, cfg CallConfig) (*EmbeddingResponse, error) {
	// Create a simple mock embedding based on text length
	// For testing purposes, create a small vector of predictable values
	textLen := float64(len(text))
	embedding := []float64{
		textLen / 100.0,  // Normalized length
		0.5,              // Fixed value
		textLen / 1000.0, // Another normalized length
	}

	return &EmbeddingResponse{
		Embedding: embedding,
		Metadata: Metadata{
			"mock":        true,
			"text_length": len(text),
		},
	}, nil
}

// reRank implements the provider interface for mock reranking
// Returns mock relevance scores for testing purposes
func (p *mockProvider) reRank(ctx context.Context, apiKey string, query string, documents []string, cfg CallConfig) (*RerankResponse, error) {
	// Create simple mock scores based on document length similarity to query
	queryLen := float64(len(query))
	scores := make([]float64, len(documents))

	for i, doc := range documents {
		docLen := float64(len(doc))
		// Score based on length similarity (0.0 to 1.0)
		// Higher score for similar length documents
		diff := queryLen - docLen
		if diff < 0 {
			diff = -diff
		}
		score := 1.0 - (diff / (queryLen + docLen + 1))
		if score < 0 {
			score = 0
		}
		scores[i] = score
	}

	return &RerankResponse{
		Scores: scores,
		Metadata: Metadata{
			"mock":      true,
			"query_len": len(query),
			"num_docs":  len(documents),
		},
	}, nil
}

// parseCompletionRequest parses an HTTP request into a CompletionRequest
// For mock provider, this accepts OpenAI format directly
func (p *mockProvider) parseCompletionRequest(req *http.Request) (*CompletionRequest, error) {
	var completionReq CompletionRequest
	if err := json.NewDecoder(req.Body).Decode(&completionReq); err != nil {
		return nil, fmt.Errorf("failed to parse mock completion request: %w", err)
	}
	return &completionReq, nil
}

// parseEmbeddingRequest parses an HTTP request into an EmbeddingRequest
// For mock provider, this accepts OpenAI format directly
func (p *mockProvider) parseEmbeddingRequest(req *http.Request) (*EmbeddingRequest, error) {
	var embeddingReq EmbeddingRequest
	if err := json.NewDecoder(req.Body).Decode(&embeddingReq); err != nil {
		return nil, fmt.Errorf("failed to parse mock embedding request: %w", err)
	}
	return &embeddingReq, nil
}

// parseRerankRequest parses an HTTP request into a RerankRequest
// For mock provider, this accepts Voyage format directly
func (p *mockProvider) parseRerankRequest(req *http.Request) (*RerankRequest, error) {
	var rerankReq RerankRequest
	if err := json.NewDecoder(req.Body).Decode(&rerankReq); err != nil {
		return nil, fmt.Errorf("failed to parse mock rerank request: %w", err)
	}
	return &rerankReq, nil
}

// buildCompletionRequest builds and executes a completion request, returning a unified response
func (p *mockProvider) buildCompletionRequest(ctx context.Context, apiKey string, req *CompletionRequest, cfg CallConfig) (*CompletionResponse, error) {
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
func (p *mockProvider) buildEmbeddingRequest(ctx context.Context, apiKey string, req *EmbeddingRequest, cfg CallConfig) (*UnifiedEmbeddingResponse, error) {
	// Create mock embedding based on text length
	textLen := float64(len(req.Input))
	embedding := []float64{
		textLen / 100.0,  // Normalized length
		0.5,              // Fixed value
		textLen / 1000.0, // Another normalized length
	}

	// Create unified response
	unifiedResp := &UnifiedEmbeddingResponse{
		Object: "list",
		Data: make([]struct {
			Object    string    `json:"object,omitempty"`
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		}, 1),
		Model: req.Model,
	}

	unifiedResp.Data[0].Object = "embedding"
	unifiedResp.Data[0].Embedding = embedding
	unifiedResp.Data[0].Index = 0

	// Add mock usage
	unifiedResp.Usage = &struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	}{
		PromptTokens: len(req.Input) / 4,
		TotalTokens:  len(req.Input) / 4,
	}

	return unifiedResp, nil
}

// buildRerankRequest builds and executes a reranking request, returning a unified response
func (p *mockProvider) buildRerankRequest(ctx context.Context, apiKey string, req *RerankRequest, cfg CallConfig) (*UnifiedRerankResponse, error) {
	// Create mock scores based on document length similarity to query
	queryLen := float64(len(req.Query))

	// Create unified response
	unifiedResp := &UnifiedRerankResponse{
		Results: make([]struct {
			Index          int     `json:"index"`
			Document       string  `json:"document,omitempty"`
			RelevanceScore float64 `json:"relevance_score"`
		}, len(req.Documents)),
		Model: req.Model,
	}

	// Calculate scores
	for i, doc := range req.Documents {
		docLen := float64(len(doc))
		diff := queryLen - docLen
		if diff < 0 {
			diff = -diff
		}
		score := 1.0 - (diff / (queryLen + docLen + 1))
		if score < 0 {
			score = 0
		}

		unifiedResp.Results[i].Index = i
		unifiedResp.Results[i].Document = doc
		unifiedResp.Results[i].RelevanceScore = score
	}

	// Add mock usage
	unifiedResp.Usage = &struct {
		TotalTokens int `json:"total_tokens,omitempty"`
	}{
		TotalTokens: len(req.Query)/4 + len(req.Documents)*10,
	}

	return unifiedResp, nil
}

// writeCompletionResponse writes a CompletionResponse as JSON to the HTTP response writer
func (p *mockProvider) writeCompletionResponse(w http.ResponseWriter, resp *CompletionResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

// writeEmbeddingResponse writes a UnifiedEmbeddingResponse as JSON to the HTTP response writer
func (p *mockProvider) writeEmbeddingResponse(w http.ResponseWriter, resp *UnifiedEmbeddingResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

// writeRerankResponse writes a UnifiedRerankResponse as JSON to the HTTP response writer
func (p *mockProvider) writeRerankResponse(w http.ResponseWriter, resp *UnifiedRerankResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}
