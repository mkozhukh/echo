package echo

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// Provider is the core interface for provider implementations: chat completions.
// Additional capabilities (embeddings, reranking, decisions) are optional and
// implemented through the embedder, reranker and decider interfaces.
type Provider interface {
	call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error)
	streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error)
}

// embedder is an optional provider capability for embeddings.
type embedder interface {
	getEmbeddings(ctx context.Context, texts []string, cfg CallConfig) (*EmbeddingResponse, error)
}

// reranker is an optional provider capability for document reranking.
type reranker interface {
	reRank(ctx context.Context, query string, documents []string, cfg CallConfig) (*RerankResponse, error)
}

// compile-time checks for provider capabilities
var (
	_ embedder = (*OpenAIProvider)(nil)
	_ embedder = (*GoogleProvider)(nil)
	_ embedder = (*VoyageProvider)(nil)
	_ reranker = (*VoyageProvider)(nil)
	_ decider  = (*TypeSafeProvider)(nil)
	_ decider  = (*MockProvider)(nil)
)

// capability returns the provider as capability T, or an error naming the provider
func capability[T any](p Provider, providerName, feature string) (T, error) {
	c, ok := p.(T)
	if !ok {
		var zero T
		return zero, fmt.Errorf("provider %q does not support %s", providerName, feature)
	}
	return c, nil
}

// CommonClient is the main client that delegates to appropriate providers
type CommonClient struct {
	apiKey      string
	baseConfig  CallConfig
	providerMap map[string]Provider
}

// NewCommonClient creates a new CommonClient instance
func NewClient(opts ...CallOption) (Client, error) {
	// Build base config with the model
	cfg := CallConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Initialize client with provider map
	client := &CommonClient{
		baseConfig:  cfg,
		providerMap: map[string]Provider{},
	}

	return client, nil
}

func (c *CommonClient) SetProvider(name string, provider Provider) {
	c.providerMap[name] = provider
}

type providerRetriver func(string) Provider

var knownProviders = map[string]providerRetriver{
	"openai":     func(key string) Provider { return &OpenAIProvider{Key: key} },
	"anthropic":  func(key string) Provider { return &AnthropicProvider{Key: key} },
	"google":     func(key string) Provider { return &GoogleProvider{Key: key} },
	"mock":       func(key string) Provider { return &MockProvider{} },
	"openrouter": func(key string) Provider { return &OpenAIProvider{Key: key} },
	"voyage":     func(key string) Provider { return &VoyageProvider{Key: key} },
	"xai":        func(key string) Provider { return &XAIProvider{Key: key} },
	"typesafe":   func(key string) Provider { return &TypeSafeProvider{Key: key} },
}

func NewCommonClient(keys map[string]string, opts ...CallOption) (Client, error) {
	client, err := NewClient(opts...)
	if err != nil {
		return nil, err
	}

	if keys == nil {
		for name, retriver := range knownProviders {
			envName := strings.ToUpper(name) + "_API_KEY"
			apiKey := os.Getenv(envName)
			if apiKey == "" {
				apiKey = os.Getenv("ECHO_KEY")
			}

			client.SetProvider(name, retriver(apiKey))
		}
	} else {
		for name, key := range keys {
			client.SetProvider(name, knownProviders[name](key))
		}
	}

	return client, nil
}

// prepareCall resolves provider, model, and configuration for a call
func (c *CommonClient) prepareCall(opts ...CallOption) (Provider, CallConfig, error) {
	_, p, cfg, err := c.resolveCall(opts...)
	return p, cfg, err
}

// resolveCall merges options and resolves provider name, provider and configuration
func (c *CommonClient) resolveCall(opts ...CallOption) (string, Provider, CallConfig, error) {
	// Merge configs
	cfg := c.baseConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	// Resolve provider and model
	providerName, resolvedModel, endpoint, err := c.resolveProviderAndModel(cfg.Model)
	if err != nil {
		return "", nil, cfg, err
	}

	// Update config with resolved model
	cfg.Model = resolvedModel
	cfg.EndPoint = endpoint

	// Get provider
	p, ok := c.providerMap[providerName]
	if !ok {
		return providerName, nil, cfg, fmt.Errorf("unknown provider: %s", providerName)
	}

	// Special handling for openrouter
	if providerName == "openrouter" {
		if cfg.BaseURL == "" {
			cfg.BaseURL = "https://openrouter.ai/api/v1/chat/completions"
		}
	}

	return providerName, p, cfg, nil
}

// prepareCapability resolves the call and returns the provider as capability T
func prepareCapability[T any](c *CommonClient, feature string, opts []CallOption) (T, CallConfig, error) {
	name, p, cfg, err := c.resolveCall(opts...)
	if err != nil {
		var zero T
		return zero, cfg, err
	}
	t, err := capability[T](p, name, feature)
	return t, cfg, err
}

// Complete implements the Client interface
func (c *CommonClient) Complete(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.call(ctx, messages, cfg)
}

// StreamComplete implements the Client interface
func (c *CommonClient) StreamComplete(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.streamCall(ctx, messages, cfg)
}

// GetEmbeddings implements the Client interface
func (c *CommonClient) GetEmbeddings(ctx context.Context, texts []string, opts ...CallOption) (*EmbeddingResponse, error) {
	e, cfg, err := prepareCapability[embedder](c, "embeddings", opts)
	if err != nil {
		return nil, err
	}
	return e.getEmbeddings(ctx, texts, cfg)
}

// ReRank implements the Client interface
func (c *CommonClient) ReRank(ctx context.Context, query string, documents []string, opts ...CallOption) (*RerankResponse, error) {
	r, cfg, err := prepareCapability[reranker](c, "rerank", opts)
	if err != nil {
		return nil, err
	}
	return r.reRank(ctx, query, documents, cfg)
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
	"openai/best":     "openai/gpt-5.2",
	"openai/balanced": "openai/gpt-5-mini",
	"openai/light":    "openai/gpt-5-nano",

	"anthropic/best":     "anthropic/claude-sonnet-4-5",
	"anthropic/balanced": "anthropic/claude-opus-4-5",
	"anthropic/light":    "anthropic/claude-haiku-4-5",

	"google/best":     "google/gemini-2.5-pro",
	"google/balanced": "google/gemini-2.5-flash",
	"google/light":    "google/gemini-2.5-flash",

	"openrouter/best":     "openrouter/openai/gpt-5",
	"openrouter/balanced": "openrouter/openai/gpt-5-mini",
	"openrouter/light":    "openrouter/openai/gpt-5-nano",

	"voyage/best":     "voyage/voyage-4-large",
	"voyage/balanced": "voyage/voyage-4",
	"voyage/light":    "voyage/voyage-4-lite",

	"xai/best":     "xai/grok-4-0709",
	"xai/balanced": "xai/grok-4-1-fast-reasoning",
	"xai/light":    "xai/grok-4-1-fast-non-reasoning",

	"typesafe/best":     "typesafe/jev-latest",
	"typesafe/balanced": "typesafe/jev-latest",
	"typesafe/light":    "typesafe/jev-latest",
}
