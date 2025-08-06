package echo

import (
	"context"
	"fmt"
	"strings"
)

type MockClient struct {
	// Mock client doesn't need any fields for basic functionality
}

func NewMockClient(authKey, model string, opts ...CallOption) *MockClient {
	return &MockClient{}
}

// Call implements the Client interface for regular (non-streaming) calls
func (c *MockClient) Call(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return nil, fmt.Errorf("invalid message chain: %w", err)
	}

	// Combine all message content
	var combinedContent strings.Builder
	for i, msg := range messages {
		if i > 0 {
			combinedContent.WriteString("\n")
		}
		combinedContent.WriteString(fmt.Sprintf("[%s]: %s", msg.Role, msg.Content))
	}

	return &Response{
		Text: combinedContent.String(),
		Metadata: Metadata{
			"mock":          true,
			"message_count": len(messages),
		},
	}, nil
}

// StreamCall implements the Client interface for streaming calls
func (c *MockClient) StreamCall(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return nil, fmt.Errorf("invalid message chain: %w", err)
	}

	// Combine all message content
	var combinedContent strings.Builder
	for i, msg := range messages {
		if i > 0 {
			combinedContent.WriteString("\n")
		}
		combinedContent.WriteString(fmt.Sprintf("[%s]: %s", msg.Role, msg.Content))
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
		content := combinedContent.String()
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
