package echo

import (
	"fmt"
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
		}
	}

	return nil
}
