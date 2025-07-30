package echo

import (
	"context"
	"fmt"
	"os"
	"strings"
)

var modelAliases = map[string]string{
	"openai/best":     "openai/gpt-4.1",
	"openai/balanced": "openai/gpt-4.1-mini",
	"openai/light":    "openai/gpt-4.1-nano",

	"anthropic/best":     "anthropic/claude-opus-4-20250514",
	"anthropic/balanced": "anthropic/claude-sonnet-4-20250514",
	"anthropic/light":    "anthropic/claude-3-5-haiku-20241022",

	"gemini/best":     "gemini/gemini-2.5-pro",
	"gemini/balanced": "gemini/gemini-2.5-flash",
	"gemini/light":    "gemini/gemini-2.5-flash",
}

// Client is the main interface for LLM operations
type Client interface {
	// Call sends a prompt and returns the response
	Call(ctx context.Context, prompt string, opts ...CallOption) (*Response, error)
	StreamCall(ctx context.Context, prompt string, opts ...CallOption) (*StreamResponse, error)
}

type Metadata = map[string]any

// Response represents the LLM response
type Response struct {
	Text     string   `json:"text"`
	Metadata Metadata `json:"metadata,omitempty"`
}

type StreamChunk struct {
	Data  string
	Meta  *Metadata // Set on first chunk if available
	Error error     // Set on error or completion
}

type StreamResponse struct {
	Stream <-chan StreamChunk
}

// CallOption allows optional parameters for calls
type CallOption func(*CallConfig)

// CallConfig holds optional call parameters
type CallConfig struct {
	BaseURL string `json:"base_url,omitempty"`
	Model   string `json:"model,omitempty"`

	Temperature *float64 `json:"temperature,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	SystemMsg   string   `json:"system_message,omitempty"`
}

func WithTemperature(temp float64) CallOption {
	return func(cfg *CallConfig) {
		cfg.Temperature = &temp
	}
}

func WithMaxTokens(tokens int) CallOption {
	return func(cfg *CallConfig) {
		cfg.MaxTokens = &tokens
	}
}

func WithSystemMessage(msg string) CallOption {
	return func(cfg *CallConfig) {
		cfg.SystemMsg = msg
	}
}

func WithModel(model string) CallOption {
	return func(cfg *CallConfig) {
		cfg.Model = model
	}
}

// NewClient creates a new LLM client based on provider/model string
func NewClient(providerModel string, apiKey string, opts ...CallOption) (Client, error) {
	if providerModel == "" {
		providerModel = os.Getenv("ECHO_MODEL")
	}

	// resolve model name
	resolvedName, ok := modelAliases[providerModel]
	if ok {
		providerModel = resolvedName
	}

	parts := strings.SplitN(providerModel, "/", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid model format: %s. Expected provider/model-name", providerModel)
	}

	provider, modelName := parts[0], parts[1]

	// Get API key from env if not provided
	if apiKey == "" {
		apiKey = os.Getenv("ECHO_KEY")
	}

	// look for more specialised env vars
	if apiKey == "" {
		envName := strings.ToUpper(provider) + "_API_KEY"
		apiKey = os.Getenv(envName)
	}

	switch provider {
	case "openai":
		return NewOpenAIClient(apiKey, modelName, opts...), nil
	case "anthropic":
		return NewAnthropicClient(apiKey, modelName, opts...), nil
	case "gemini":
		return NewGeminiClient(apiKey, modelName, opts...), nil
	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}
}
