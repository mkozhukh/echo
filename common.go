package echo

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// provider interface for internal provider implementations
type provider interface {
	call(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*Response, error)
	streamCall(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*StreamResponse, error)
	getEmbeddings(ctx context.Context, apiKey string, text string, cfg CallConfig) (*EmbeddingResponse, error)
}

// CommonClient is the main client that delegates to appropriate providers
type CommonClient struct {
	apiKey      string
	baseConfig  CallConfig
	providerMap map[string]provider
}

// NewCommonClient creates a new CommonClient instance
func NewCommonClient(fullModelName string, apiKey string, opts ...CallOption) (*CommonClient, error) {
	if fullModelName == "" {
		fullModelName = os.Getenv("ECHO_MODEL")
	}

	if fullModelName != "" {
		// Parse initial model just to check if it's valid
		_, _, _, err := parseModelString(fullModelName)
		if err != nil {
			return nil, err
		}
	}

	// Build base config with the model
	cfg := CallConfig{
		Model: fullModelName,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Initialize client with provider map
	client := &CommonClient{
		apiKey:     apiKey,
		baseConfig: cfg,
		providerMap: map[string]provider{
			"openai":     &openAIProvider{},
			"anthropic":  &anthropicProvider{},
			"google":     &googleProvider{},
			"mock":       &mockProvider{},
			"openrouter": &openAIProvider{}, // OpenRouter uses OpenAI API
			"voyage":     &voyageProvider{}, // Voyage AI - embeddings only
		},
	}

	return client, nil
}

// prepareCall resolves provider, model, and configuration for a call
func (c *CommonClient) prepareCall(opts ...CallOption) (provider, string, CallConfig, error) {
	// Merge configs
	cfg := c.baseConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	// Resolve provider and model
	providerName, resolvedModel, endpoint, err := c.resolveProviderAndModel(cfg.Model)
	if err != nil {
		return nil, "", cfg, err
	}

	// Resolve API key
	apiKey := c.resolveAPIKey(c.apiKey, providerName)

	// Update config with resolved model
	cfg.Model = resolvedModel
	cfg.EndPoint = endpoint

	// Get provider
	p, ok := c.providerMap[providerName]
	if !ok {
		return nil, "", cfg, fmt.Errorf("unknown provider: %s", providerName)
	}

	// Special handling for openrouter
	if providerName == "openrouter" {
		if cfg.BaseURL == "" {
			cfg.BaseURL = "https://openrouter.ai/api/v1/chat/completions"
		}
	}

	return p, apiKey, cfg, nil
}

// Call implements the Client interface
func (c *CommonClient) Call(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error) {
	p, apiKey, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.call(ctx, apiKey, messages, cfg)
}

// StreamCall implements the Client interface
func (c *CommonClient) StreamCall(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error) {
	p, apiKey, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.streamCall(ctx, apiKey, messages, cfg)
}

// GetEmbeddings implements the Client interface
func (c *CommonClient) GetEmbeddings(ctx context.Context, text string, opts ...CallOption) (*EmbeddingResponse, error) {
	p, apiKey, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.getEmbeddings(ctx, apiKey, text, cfg)
}

func (c *CommonClient) resolveAPIKey(apiKey string, providerName string) string {
	if apiKey == "" {
		envName := strings.ToUpper(providerName) + "_API_KEY"
		apiKey = os.Getenv(envName)
	}
	if apiKey == "" {
		apiKey = os.Getenv("ECHO_KEY")
	}
	return apiKey
}

// resolveProviderAndModel determines the provider and resolves model aliases
func (c *CommonClient) resolveProviderAndModel(modelStr string) (string, string, string, error) {
	// Use override model if provided, otherwise use base config model
	if modelStr == "" {
		modelStr = c.baseConfig.Model
	}
	if modelStr == "" {
		modelStr = os.Getenv("ECHO_MODEL")
	}
	if modelStr == "" {
		return "", "", "", fmt.Errorf("no model specified")
	}

	resolvedModel, ok := alises[modelStr]
	if ok {
		modelStr = resolvedModel
	}

	providerName, modelName, endpoint, err := parseModelString(modelStr)
	if err != nil {
		return "", "", "", err
	}

	return providerName, modelName, endpoint, nil
}

// parseModelString parses "provider/model@endpoint" format
func parseModelString(fullModelName string) (string, string, string, error) {
	parts := strings.SplitN(fullModelName, "/", 2)
	if len(parts) != 2 {
		return "", "", "", fmt.Errorf("invalid model format: %s. Expected provider/model-name@endpoint", fullModelName)
	}

	provider, modelName := parts[0], parts[1]
	endpoint := ""

	// Remove endpoint suffix if present (handled in CallConfig.EndPoint)
	if atIndex := strings.Index(modelName, "@"); atIndex != -1 {
		endpoint = modelName[atIndex+1:]
		modelName = modelName[:atIndex]
	}

	return provider, modelName, endpoint, nil
}

// Model aliases for each provider
var alises = map[string]string{
	"openai/best":     "openai/gpt-5",
	"openai/balanced": "openai/gpt-5-mini",
	"openai/light":    "openai/gpt-5-nano",

	"anthropic/best":     "anthropic/claude-opus-4-1-20250805",
	"anthropic/balanced": "anthropic/claude-sonnet-4-20250514",
	"anthropic/light":    "anthropic/claude-3-5-haiku-20241022",

	"google/best":     "google/gemini-2.5-pro",
	"google/balanced": "google/gemini-2.5-flash",
	"google/light":    "google/gemini-2.5-flash",

	"openrouter/best":     "openrouter/openai/gpt-5",
	"openrouter/balanced": "openrouter/openai/gpt-5-mini",
	"openrouter/light":    "openrouter/openai/gpt-5-nano",

	"voyage/best":     "voyage/voyage-3",
	"voyage/balanced": "voyage/voyage-3-lite",
	"voyage/light":    "voyage/voyage-3-lite",
}
