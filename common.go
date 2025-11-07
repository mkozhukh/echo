package echo

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
)

// provider interface for internal provider implementations
type Provider interface {
	call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error)
	streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error)
	getEmbeddings(ctx context.Context, text string, cfg CallConfig) (*EmbeddingResponse, error)
	reRank(ctx context.Context, query string, documents []string, cfg CallConfig) (*RerankResponse, error)

	// Parse HTTP requests into unified request structures
	parseCompletionRequest(req *http.Request) (*CompletionRequest, error)
	parseEmbeddingRequest(req *http.Request) (*EmbeddingRequest, error)
	parseRerankRequest(req *http.Request) (*RerankRequest, error)

	// Build methods - consume parsed requests and return unified responses
	buildCompletionRequest(ctx context.Context, req *CompletionRequest, cfg CallConfig) (*CompletionResponse, error)
	buildEmbeddingRequest(ctx context.Context, req *EmbeddingRequest, cfg CallConfig) (*UnifiedEmbeddingResponse, error)
	buildRerankRequest(ctx context.Context, req *RerankRequest, cfg CallConfig) (*UnifiedRerankResponse, error)

	// Write methods - write unified responses back as HTTP responses
	writeCompletionResponse(w http.ResponseWriter, resp *CompletionResponse) error
	writeEmbeddingResponse(w http.ResponseWriter, resp *UnifiedEmbeddingResponse) error
	writeRerankResponse(w http.ResponseWriter, resp *UnifiedRerankResponse) error
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
	// Merge configs
	cfg := c.baseConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	// Resolve provider and model
	providerName, resolvedModel, endpoint, err := c.resolveProviderAndModel(cfg.Model)
	if err != nil {
		return nil, cfg, err
	}

	// Update config with resolved model
	cfg.Model = resolvedModel
	cfg.EndPoint = endpoint

	// Get provider
	p, ok := c.providerMap[providerName]
	if !ok {
		return nil, cfg, fmt.Errorf("unknown provider: %s", providerName)
	}

	// Special handling for openrouter
	if providerName == "openrouter" {
		if cfg.BaseURL == "" {
			cfg.BaseURL = "https://openrouter.ai/api/v1/chat/completions"
		}
	}

	return p, cfg, nil
}

// prepareCall resolves provider, model, and configuration for a call
func (c *CommonClient) getProvider(opts ...CallOption) (Provider, error) {
	// Merge configs
	cfg := c.baseConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	// Resolve provider and model
	providerName, _, _, err := c.resolveProviderAndModel(cfg.Model)
	if err != nil {
		return nil, err
	}

	// Get provider
	p, ok := c.providerMap[providerName]
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", providerName)
	}

	return p, nil
}

// Call implements the Client interface
func (c *CommonClient) Complete(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.call(ctx, messages, cfg)
}

// StreamCall implements the Client interface
func (c *CommonClient) StreamComplete(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.streamCall(ctx, messages, cfg)
}

// GetEmbeddings implements the Client interface
func (c *CommonClient) GetEmbeddings(ctx context.Context, text string, opts ...CallOption) (*EmbeddingResponse, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.getEmbeddings(ctx, text, cfg)
}

// ReRank implements the Client interface
func (c *CommonClient) ReRank(ctx context.Context, query string, documents []string, opts ...CallOption) (*RerankResponse, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.reRank(ctx, query, documents, cfg)
}

func (c *CommonClient) ParseComplete(req *http.Request, opts ...CallOption) (*CompletionRequest, error) {
	p, err := c.getProvider(opts...)
	if err != nil {
		return nil, err
	}
	return p.parseCompletionRequest(req)
}

func (c *CommonClient) ExecComplete(ctx context.Context, CompletionRequest *CompletionRequest, opts ...CallOption) (*CompletionResponse, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.buildCompletionRequest(ctx, CompletionRequest, cfg)
}

func (c *CommonClient) WriteComplete(w http.ResponseWriter, resp *CompletionResponse, opts ...CallOption) error {
	p, err := c.getProvider(opts...)
	if err != nil {
		return err
	}
	return p.writeCompletionResponse(w, resp)
}

func (c *CommonClient) ParseEmbedding(req *http.Request, opts ...CallOption) (*EmbeddingRequest, error) {
	p, err := c.getProvider(opts...)
	if err != nil {
		return nil, err
	}
	return p.parseEmbeddingRequest(req)
}

func (c *CommonClient) ExecEmbedding(ctx context.Context, EmbeddingRequest *EmbeddingRequest, opts ...CallOption) (*UnifiedEmbeddingResponse, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.buildEmbeddingRequest(ctx, EmbeddingRequest, cfg)
}

func (c *CommonClient) WriteEmbedding(w http.ResponseWriter, resp *UnifiedEmbeddingResponse, opts ...CallOption) error {
	p, err := c.getProvider(opts...)
	if err != nil {
		return err
	}
	return p.writeEmbeddingResponse(w, resp)
}

func (c *CommonClient) ParseRerank(req *http.Request, opts ...CallOption) (*RerankRequest, error) {
	p, err := c.getProvider(opts...)
	if err != nil {
		return nil, err
	}
	return p.parseRerankRequest(req)
}

func (c *CommonClient) ExecRerank(ctx context.Context, RerankRequest *RerankRequest, opts ...CallOption) (*UnifiedRerankResponse, error) {
	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}
	return p.buildRerankRequest(ctx, RerankRequest, cfg)
}

func (c *CommonClient) WriteRerank(w http.ResponseWriter, resp *UnifiedRerankResponse, opts ...CallOption) error {
	p, err := c.getProvider(opts...)
	if err != nil {
		return err
	}
	return p.writeRerankResponse(w, resp)
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
