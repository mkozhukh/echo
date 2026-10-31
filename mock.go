package echo

import (
	"context"
	"fmt"
	"sort"
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

	responseText := p.getMessages(messages, cfg)

	// If structured output is requested, return mock JSON
	if cfg.StructuredOutput != nil {
		responseText = fmt.Sprintf(`{"mock_response": true, "schema_name": %q}`,
			cfg.StructuredOutput.Name)
	}

	return &Response{
		Text: responseText,
		Metadata: Metadata{
			"mock":              true,
			"message_count":     len(messages),
			"structured_output": cfg.StructuredOutput != nil,
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
				"mock":              true,
				"message_count":     len(messages),
				"structured_output": cfg.StructuredOutput != nil,
			},
		}

		// Simulate streaming by sending the combined content in chunks
		var content string
		if cfg.StructuredOutput != nil {
			content = fmt.Sprintf(`{"mock_response": true, "schema_name": %q}`,
				cfg.StructuredOutput.Name)
		} else {
			content = p.getMessages(messages, cfg)
		}
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

// decide implements the decider interface with deterministic answers:
// choice picks the first option, score picks the lowest level, noul returns 0.5.
func (p *MockProvider) decide(ctx context.Context, req DecisionRequest, cfg CallConfig) (*DecisionResponse, error) {
	one := 1.0
	answers := make(map[string]Answer, len(req.Questions))

	for id, q := range req.Questions {
		a := Answer{Type: q.Type}
		switch q.Type {
		case QuestionChoice:
			names := mockOptionNames(q.Criteria)
			if len(names) == 0 {
				return nil, fmt.Errorf("question %q: no options", id)
			}
			a.Choice = names[0]
			a.Confidence = &one
			a.Probabilities = map[string]float64{}
			for i, n := range names {
				if i == 0 {
					a.Probabilities[n] = 1
				} else {
					a.Probabilities[n] = 0
				}
			}
		case QuestionScore:
			levels, _ := q.Criteria.([]string)
			a.Score = 0
			a.Confidence = &one
			a.Legend = map[string]string{}
			a.Probabilities = map[string]float64{}
			for i, l := range levels {
				key := fmt.Sprint(i)
				a.Legend[key] = l
				if i == 0 {
					a.Probabilities[key] = 1
				} else {
					a.Probabilities[key] = 0
				}
			}
		case QuestionNoul:
			a.Noul = 0.5
		}
		answers[id] = a
	}

	return &DecisionResponse{
		Model:   req.Model,
		Answers: answers,
		Metadata: Metadata{
			"mock":           true,
			"question_count": len(req.Questions),
		},
	}, nil
}

func mockOptionNames(criteria any) []string {
	switch c := criteria.(type) {
	case choiceCriteria:
		names := make([]string, len(c))
		for i, o := range c {
			names[i] = o.Name
		}
		return names
	case []Option:
		return mockOptionNames(choiceCriteria(c))
	case []string:
		return c
	case map[string]string:
		names := make([]string, 0, len(c))
		for n := range c {
			names = append(names, n)
		}
		sort.Strings(names)
		return names
	}
	return nil
}
