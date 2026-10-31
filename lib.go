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

// EmbeddingResponse represents the embedding response
type EmbeddingResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Metadata   Metadata    `json:"metadata,omitempty"`
}

// RerankResponse represents the rerank response
type RerankResponse struct {
	Scores   []float32 `json:"scores"`
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
