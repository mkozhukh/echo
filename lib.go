package echo

import (
	"context"
)

// Client is the main interface for LLM operations
type Client interface {
	// Call sends a message chain and returns the response
	Call(ctx context.Context, messages []Message, opts ...CallOption) (*Response, error)
	StreamCall(ctx context.Context, messages []Message, opts ...CallOption) (*StreamResponse, error)
	// GetEmbeddings calculates embeddings for the given text
	GetEmbeddings(ctx context.Context, text string, opts ...CallOption) (*EmbeddingResponse, error)
	// ReRank reranks documents based on relevance to query
	ReRank(ctx context.Context, query string, documents []string, opts ...CallOption) (*RerankResponse, error)
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
	Embedding []float64 `json:"embedding"`
	Metadata  Metadata  `json:"metadata,omitempty"`
}

// RerankResponse represents the rerank response
type RerankResponse struct {
	Scores   []float64 `json:"scores"`
	Metadata Metadata  `json:"metadata,omitempty"`
}

// Unified request structures for parsing HTTP requests
// Using OpenAI format as the common format to minimize data copying

// CompletionRequest represents a unified completion request
// Based on OpenAI's chat completion format
type CompletionRequest struct {
	Model         string          `json:"model"`
	Temperature   *float64        `json:"temperature,omitempty"`
	MaxTokens     *int            `json:"max_completion_tokens,omitempty"`
	Messages      []OpenAIMessage `json:"messages"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
}

// OpenAIMessage represents a message in OpenAI format
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// EmbeddingRequest represents a unified embedding request
// Based on OpenAI's embedding format
type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// RerankRequest represents a unified reranking request
// Based on Voyage AI's rerank format
type RerankRequest struct {
	Query      string   `json:"query"`
	Documents  []string `json:"documents"`
	Model      string   `json:"model"`
	TopK       *int     `json:"top_k,omitempty"`
	Truncation *bool    `json:"truncation,omitempty"`
}

// Unified response structures for Build methods
// Using OpenAI format as the common format to minimize data copying

// CompletionResponse represents a unified completion response
// Based on OpenAI's chat completion format
type CompletionResponse struct {
	ID      string `json:"id,omitempty"`
	Object  string `json:"object,omitempty"`
	Created int64  `json:"created,omitempty"`
	Model   string `json:"model,omitempty"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

// UnifiedEmbeddingResponse represents a unified embedding response
// Based on OpenAI's embedding format
type UnifiedEmbeddingResponse struct {
	Object string `json:"object,omitempty"`
	Data   []struct {
		Object    string    `json:"object,omitempty"`
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model,omitempty"`
	Usage *struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

// UnifiedRerankResponse represents a unified reranking response
// Based on Voyage AI's rerank format
type UnifiedRerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		Document       string  `json:"document,omitempty"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
	Model string `json:"model,omitempty"`
	Usage *struct {
		TotalTokens int `json:"total_tokens,omitempty"`
	} `json:"usage,omitempty"`
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
	return NewCommonClient(fullModelName, apiKey, opts...)
}
