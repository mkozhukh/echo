package echo

import (
	"fmt"
	"testing"
)

func TestTemplateMessage(t *testing.T) {
	// Test basic template parsing
	template := `
@system:
System prompt

@agent:
Follow the task

@user:
Write a poem
`

	messages := TemplateMessage(template)

	// Should have 3 messages
	if len(messages) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(messages))
	}

	// Check first message
	if messages[0].Role != System || messages[0].Content != "System prompt" {
		t.Errorf("First message incorrect: %+v", messages[0])
	}

	// Check second message
	if messages[1].Role != Agent || messages[1].Content != "Follow the task" {
		t.Errorf("Second message incorrect: %+v", messages[1])
	}

	// Check third message
	if messages[2].Role != User || messages[2].Content != "Write a poem" {
		t.Errorf("Third message incorrect: %+v", messages[2])
	}
}

func TestTemplateMessageMultiline(t *testing.T) {
	// Test multiline content
	template := `@system:
You are a helpful assistant.
You always respond politely.

@user:
Hello there!
How are you?`

	messages := TemplateMessage(template)

	if len(messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(messages))
	}

	expectedSystem := "You are a helpful assistant.\nYou always respond politely."
	if messages[0].Content != expectedSystem {
		t.Errorf("System content incorrect:\nExpected: %q\nGot: %q", expectedSystem, messages[0].Content)
	}

	expectedUser := "Hello there!\nHow are you?"
	if messages[1].Content != expectedUser {
		t.Errorf("User content incorrect:\nExpected: %q\nGot: %q", expectedUser, messages[1].Content)
	}
}

func TestTemplateMessageInlineContent(t *testing.T) {
	// Test content on same line as role marker
	template := `@system: You are helpful
@user: Hello`

	messages := TemplateMessage(template)

	if len(messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(messages))
	}

	if messages[0].Content != "You are helpful" {
		t.Errorf("System content incorrect: %q", messages[0].Content)
	}

	if messages[1].Content != "Hello" {
		t.Errorf("User content incorrect: %q", messages[1].Content)
	}
}

func ExampleTemplateMessage() {
	template := `
@system:
You are a helpful math tutor.

@user:
What is 2+2?

@agent:
2+2 equals 4.

@user:
Can you explain why?
`

	messages := TemplateMessage(template)

	for _, msg := range messages {
		fmt.Printf("%s: %s\n", msg.Role, msg.Content)
	}
	// Output:
	// system: You are a helpful math tutor.
	// user: What is 2+2?
	// agent: 2+2 equals 4.
	// user: Can you explain why?
}
