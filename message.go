package echo

import (
	"fmt"
	"strings"
)

const (
	System = "system"
	Agent  = "agent"
	User   = "user"
)

type Message struct {
	Content string
	Role    string
}

// QuickMessage creates a simple user message chain for backward compatibility
func QuickMessage(message string) []Message {
	return []Message{
		{
			Role:    User,
			Content: message,
		},
	}
}

// validateMessages validates the message chain according to the rules:
// - Must not be empty
// - System message (if present) must be first
// - Only one system message allowed
// - Roles must be valid (system, user, agent)
func validateMessages(messages []Message) error {
	if len(messages) == 0 {
		return fmt.Errorf("message chain cannot be empty")
	}

	systemMessageSeen := false
	userMessageSeen := false
	for i, msg := range messages {
		// Validate role
		switch msg.Role {
		case System, User, Agent:
			// Valid roles
		default:
			return fmt.Errorf("invalid role '%s' at position %d", msg.Role, i)
		}

		// Check system message rules
		if msg.Role == System {
			if i > 0 {
				return fmt.Errorf("system message must be first in the chain")
			}
			if systemMessageSeen {
				return fmt.Errorf("only one system message allowed")
			}
			systemMessageSeen = true
		} else {
			userMessageSeen = true
		}
	}

	if !userMessageSeen {
		return fmt.Errorf("at least one non-system message is required")
	}
	return nil
}

// TemplateMessage parses a template string into a message chain.
// The template format uses @role: markers to separate messages.
// Example:
//
//	@system:
//	You are a helpful assistant
//	@user:
//	Hello
func TemplateMessage(template string) []Message {
	messages := []Message{}
	lines := strings.Split(template, "\n")

	var currentRole string
	var contentLines []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Check if this line starts a new section
		if strings.HasPrefix(trimmed, "@") && strings.Contains(trimmed, ":") {
			// Save previous section if exists
			if currentRole != "" && len(contentLines) > 0 {
				content := strings.TrimSpace(strings.Join(contentLines, "\n"))
				if content != "" {
					messages = append(messages, Message{
						Role:    currentRole,
						Content: content,
					})
				}
			}

			// Parse new role
			parts := strings.SplitN(trimmed, ":", 2)
			roleStr := strings.TrimPrefix(parts[0], "@")
			currentRole = strings.TrimSpace(roleStr)

			// Validate role
			switch currentRole {
			case "system":
				currentRole = System
			case "user":
				currentRole = User
			case "agent":
				currentRole = Agent
			default:
				// Skip invalid roles
				currentRole = ""
			}

			// Reset content for new section
			contentLines = []string{}

			// If there's content on the same line after the colon, add it
			if len(parts) > 1 && strings.TrimSpace(parts[1]) != "" {
				contentLines = append(contentLines, strings.TrimSpace(parts[1]))
			}
		} else if currentRole != "" {
			// Add line to current section
			contentLines = append(contentLines, line)
		}
	}

	// Save last section if exists
	if currentRole != "" && len(contentLines) > 0 {
		content := strings.TrimSpace(strings.Join(contentLines, "\n"))
		if content != "" {
			messages = append(messages, Message{
				Role:    currentRole,
				Content: content,
			})
		}
	}

	return messages
}
