package echo

import (
	"context"
)

// Client is the main interface for LLM operations
type Client interface {
	// SetProvider sets a provider for the client
	SetProvider(name string, provider Provider)
	// Complete sends a message chain and returns the response
	Complete(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error)
	// StreamComplete sends a message chain and returns the response as a stream
	StreamComplete(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error)
	// GetEmbeddings calculates embeddings for the given texts
	GetEmbeddings(ctx context.Context, texts []string, opts ...CallOption) (*EmbeddingResponse, error)
	// ReRank reranks documents based on relevance to query
	ReRank(ctx context.Context, query string, documents []string, opts ...CallOption) (*RerankResponse, error)
	// Decide evaluates typed questions (choice, score, noul) against the state
	// and returns structured answers. Supported by decision models (e.g. typesafe/jev-latest).
	Decide(ctx context.Context, state any, questions Questions, opts ...CallOption) (*DecisionResponse, error)
}

type Metadata = map[string]any

// Usage reports token consumption of a call, normalized across providers.
// Prices are not computed by the library; CostUSD is set only when the
// provider reports the charged amount directly (e.g. OpenRouter).
type Usage struct {
	Model             string   `json:"model,omitempty"`               // resolved "provider/model" used for the call
	InputTokens       int      `json:"input_tokens"`                  // all input tokens, including cached ones
	CachedInputTokens int      `json:"cached_input_tokens,omitempty"` // input tokens read from cache (subset of InputTokens)
	CacheWriteTokens  int      `json:"cache_write_tokens,omitempty"`  // input tokens written to cache (subset of InputTokens)
	OutputTokens      int      `json:"output_tokens"`                 // all output tokens, including reasoning
	ReasoningTokens   int      `json:"reasoning_tokens,omitempty"`    // reasoning/thinking tokens (subset of OutputTokens)
	TotalTokens       int      `json:"total_tokens"`
	CostUSD           *float64 `json:"cost_usd,omitempty"` // provider-reported cost, if any
}

// Add accumulates other into u. Safe to call with a nil other.
// Model is kept only while all accumulated usages share the same model;
// price each call separately when mixing models.
func (u *Usage) Add(other *Usage) {
	if other == nil {
		return
	}
	if u.isEmpty() {
		u.Model = other.Model
	} else if u.Model != other.Model {
		u.Model = ""
	}
	u.InputTokens += other.InputTokens
	u.CachedInputTokens += other.CachedInputTokens
	u.CacheWriteTokens += other.CacheWriteTokens
	u.OutputTokens += other.OutputTokens
	u.ReasoningTokens += other.ReasoningTokens
	u.TotalTokens += other.TotalTokens
	if other.CostUSD != nil {
		cost := *other.CostUSD
		if u.CostUSD != nil {
			cost += *u.CostUSD
		}
		u.CostUSD = &cost
	}
}

func (u *Usage) isEmpty() bool {
	return u.Model == "" && u.InputTokens == 0 && u.OutputTokens == 0 && u.TotalTokens == 0 && u.CostUSD == nil
}

// Pricer computes the cost of a call from its usage.
// It is implemented by pricing.Prices; see WithPricer.
type Pricer interface {
	// USD returns the cost in USD and whether a price is known for u.Model
	USD(u Usage) (float64, bool)
}

// Response represents the LLM response
type Response struct {
	Text     string   `json:"text"`
	Usage    *Usage   `json:"usage,omitempty"` // nil when the provider does not report usage
	Metadata Metadata `json:"metadata,omitempty"`
}

type StreamChunk struct {
	Data  string
	Usage *Usage    // Set on a dedicated chunk near the end of the stream, if reported
	Meta  *Metadata // Provider-specific metadata (non-token), if any
	Error error     // Set on error or completion
}

type StreamResponse struct {
	Stream <-chan StreamChunk
}

// EmbeddingResponse represents the embedding response
type EmbeddingResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Usage      *Usage      `json:"usage,omitempty"`
	Metadata   Metadata    `json:"metadata,omitempty"`
}

// RerankResponse represents the rerank response
type RerankResponse struct {
	Scores   []float32 `json:"scores"`
	Usage    *Usage    `json:"usage,omitempty"`
	Metadata Metadata  `json:"metadata,omitempty"`
}

// CallOption allows optional parameters for calls
type CallOption func(*CallConfig)

// StructuredOutputConfig holds configuration for structured JSON output
type StructuredOutputConfig struct {
	Name   string // Name/identifier for the schema (required by OpenAI, Anthropic)
	Schema any    // JSON Schema as map[string]any
}

// CallConfig holds optional call parameters
type CallConfig struct {
	BaseURL  string
	Model    string
	EndPoint string

	Temperature      *float32
	MaxTokens        *int
	SystemMsg        string
	StructuredOutput *StructuredOutputConfig
	ReasoningEffort  string // "low", "medium", "high" - controls thinking/reasoning level
	StoreData        *bool  // xAI: set to false to disable server-side storage (default: false)
	Pricer           Pricer // fills Usage.CostUSD when the provider does not report it
}

func WithTemperature(temp float32) CallOption {
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

// WithStructuredOutput enables structured JSON output conforming to the provided schema.
// The name parameter identifies the schema (required by OpenAI and Anthropic).
// The schema parameter should be a JSON Schema as map[string]any.
func WithStructuredOutput(name string, schema any) CallOption {
	return func(cfg *CallConfig) {
		cfg.StructuredOutput = &StructuredOutputConfig{
			Name:   name,
			Schema: schema,
		}
	}
}

// WithReasoningEffort controls the thinking/reasoning level for models that support it.
// Valid values: "low", "medium", "high"
// - OpenAI: uses reasoning_effort parameter (for o1 models)
// - Anthropic: uses output_config.effort (claude-opus-4-5)
// - Google: uses thinkingConfig.thinkingLevel (gemini-3 models)
func WithReasoningEffort(effort string) CallOption {
	return func(cfg *CallConfig) {
		cfg.ReasoningEffort = effort
	}
}

// WithStoreData controls whether the provider stores conversation data on the server.
// Currently only supported by xAI (Grok) - set to false to disable server-side storage.
// Default is false for xAI to prioritize privacy.
func WithStoreData(store bool) CallOption {
	return func(cfg *CallConfig) {
		cfg.StoreData = &store
	}
}

// WithPricer makes the client fill Usage.CostUSD of every response (including
// the usage chunk of streams) using the given pricer, e.g. pricing.Default().
// A cost reported by the provider itself (OpenRouter) is kept as is.
func WithPricer(p Pricer) CallOption {
	return func(cfg *CallConfig) {
		cfg.Pricer = p
	}
}
