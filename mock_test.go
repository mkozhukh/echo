package echo

import (
	"context"
	"strings"
	"testing"
)

func TestMockClient_Call(t *testing.T) {
	client, err := NewClient("mock/test", "")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	ctx := context.Background()

	tests := []struct {
		name     string
		messages []Message
		want     string
	}{
		{
			name: "single user message",
			messages: []Message{
				{Role: User, Content: "Hello"},
			},
			want: "[user]: Hello",
		},
		{
			name: "system and user messages",
			messages: []Message{
				{Role: System, Content: "You are a helpful assistant"},
				{Role: User, Content: "Hello"},
			},
			want: "[system]: You are a helpful assistant\n[user]: Hello",
		},
		{
			name: "multiple messages",
			messages: []Message{
				{Role: System, Content: "You are a helpful assistant"},
				{Role: User, Content: "Hello"},
				{Role: Agent, Content: "Hi there!"},
				{Role: User, Content: "How are you?"},
			},
			want: "[system]: You are a helpful assistant\n[user]: Hello\n[agent]: Hi there!\n[user]: How are you?",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Call(ctx, tt.messages)
			if err != nil {
				t.Fatalf("Call() error = %v", err)
			}

			if resp.Text != tt.want {
				t.Errorf("Call() = %v, want %v", resp.Text, tt.want)
			}

			// Check metadata
			if resp.Metadata["mock"] != true {
				t.Errorf("Expected mock metadata to be true")
			}
			if resp.Metadata["message_count"] != len(tt.messages) {
				t.Errorf("Expected message_count to be %d, got %v", len(tt.messages), resp.Metadata["message_count"])
			}
		})
	}
}

func TestMockClient_StreamCall(t *testing.T) {
	client, err := NewClient("mock/test", "")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	ctx := context.Background()

	messages := []Message{
		{Role: System, Content: "You are a helpful assistant"},
		{Role: User, Content: "Hello"},
		{Role: Agent, Content: "Hi there!"},
	}

	expected := "[system]: You are a helpful assistant\n[user]: Hello\n[agent]: Hi there!"

	streamResp, err := client.StreamCall(ctx, messages)
	if err != nil {
		t.Fatalf("StreamCall() error = %v", err)
	}

	var receivedData strings.Builder
	var metadata *Metadata
	var completionError error

	for chunk := range streamResp.Stream {
		if chunk.Error != nil {
			completionError = chunk.Error
			break
		}
		if chunk.Meta != nil {
			metadata = chunk.Meta
		}
		if chunk.Data != "" {
			receivedData.WriteString(chunk.Data)
		}
	}

	if completionError != nil {
		t.Errorf("Unexpected error in stream: %v", completionError)
	}

	if receivedData.String() != expected {
		t.Errorf("StreamCall() = %v, want %v", receivedData.String(), expected)
	}

	// Check metadata
	if metadata == nil {
		t.Errorf("Expected metadata to be present")
	} else {
		if (*metadata)["mock"] != true {
			t.Errorf("Expected mock metadata to be true")
		}
		if (*metadata)["message_count"] != len(messages) {
			t.Errorf("Expected message_count to be %d, got %v", len(messages), (*metadata)["message_count"])
		}
	}
}

func TestMockClient_InvalidMessages(t *testing.T) {
	client, err := NewClient("mock/test", "")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	ctx := context.Background()

	// Test empty messages
	_, err = client.Call(ctx, []Message{})
	if err == nil {
		t.Errorf("Expected error for empty messages")
	}

	// Test invalid role
	_, err = client.Call(ctx, []Message{
		{Role: "invalid", Content: "test"},
	})
	if err == nil {
		t.Errorf("Expected error for invalid role")
	}

	// Test system message not first
	_, err = client.Call(ctx, []Message{
		{Role: User, Content: "test"},
		{Role: System, Content: "test"},
	})
	if err == nil {
		t.Errorf("Expected error for system message not first")
	}
}
