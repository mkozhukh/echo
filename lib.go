package echo

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// Client is the main interface for LLM operations
type Client interface {
	// Call sends a message chain and returns the response
	Call(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error)
	StreamCall(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error)
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
	BaseURL  string
	Model    string
	EndPoint string

	Temperature *float64
	MaxTokens   *int
	SystemMsg   string
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

func WithBaseURL(url string) CallOption {
	return func(cfg *CallConfig) {
		cfg.BaseURL = url
	}
}

func WithEndPoint(endpoint string) CallOption {
	return func(cfg *CallConfig) {
		cfg.EndPoint = endpoint
	}
}

// NewClient creates a new LLM client based on provider/model string
func NewClient(fullModelName string, apiKey string, opts ...CallOption) (Client, error) {
	if fullModelName == "" {
		fullModelName = os.Getenv("ECHO_MODEL")
	}

	parts := strings.SplitN(fullModelName, "/", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid model format: %s. Expected provider/model-name@endpoint", fullModelName)
	}

	provider, modelName := parts[0], parts[1]
	endpoint := ""

	// Check if model name contains @ for provider override
	if atIndex := strings.Index(modelName, "@"); atIndex != -1 {
		endpoint = modelName[atIndex+1:]
		modelName = modelName[:atIndex]
	}

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
	case "mock":
		return NewMockClient(apiKey, modelName, opts...), nil
	case "openai":
		return NewOpenAIClient(apiKey, modelName, opts...), nil
	case "anthropic":
		return NewAnthropicClient(apiKey, modelName, opts...), nil
	case "google":
		return NewGoogleClient(apiKey, modelName, opts...), nil
	case "openrouter":
		customOpts := append(opts, WithBaseURL("https://openrouter.ai/api/v1/chat/completions"))
		if endpoint != "" {
			customOpts = append(customOpts, WithEndPoint(endpoint))
		}
		return NewOpenAIClient(apiKey, modelName, customOpts...), nil
	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}
}
