package echo

import (
	"context"
	"fmt"
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
