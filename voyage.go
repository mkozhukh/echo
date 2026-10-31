package echo

import (
	"context"
	"fmt"
	"net/http"
	"strings"
)

// VoyageProvider is a stateless provider for Voyage AI embeddings
// Voyage AI is Anthropic's recommended embedding provider
type VoyageProvider struct {
	Key string
}

// NewVoyageClient creates a new Voyage AI client
func NewVoyageClient(apiKey, model string, opts ...CallOption) Client {
	if model != "" {
		if !strings.HasPrefix(model, "voyage/") {
			model = "voyage/" + model
		}
		opts = append([]CallOption{WithModel(model)}, opts...)
	}
	client, _ := NewClient(opts...)
	client.SetProvider("voyage", &VoyageProvider{Key: apiKey})
	return client
}

// Voyage AI structures
type voyageEmbeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type voyageError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type voyageEmbeddingResponse struct {
	Error *voyageError `json:"error,omitempty"`
	Data  []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage *struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

type voyageRerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	Model     string   `json:"model"`
	// TopK       *int     `json:"top_k,omitempty"`
	// Truncation *bool    `json:"truncation,omitempty"`
}

type voyageRerankResponse struct {
	Error   *voyageError `json:"error,omitempty"`
	Results []struct {
		Index          int     `json:"index"`
		Document       string  `json:"document"`
		RelevanceScore float32 `json:"relevance_score"`
	} `json:"data"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
	Model string `json:"model"`
}

// call implements the provider interface but returns an error
// Voyage AI only supports embeddings, not chat completions
func (p *VoyageProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
	return nil, fmt.Errorf("Voyage AI only supports embeddings, not chat completions. Use GetEmbeddings() instead")
}

// streamCall implements the provider interface but returns an error
// Voyage AI only supports embeddings, not chat completions
func (p *VoyageProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	return nil, fmt.Errorf("Voyage AI only supports embeddings, not chat completions. Use GetEmbeddings() instead")
}

// getEmbeddings implements the provider interface for Voyage AI embeddings
func (p *VoyageProvider) getEmbeddings(ctx context.Context, texts []string, cfg CallConfig) (*EmbeddingResponse, error) {
	// Use provided model or default to voyage-3
	model := cfg.Model
	if model == "" {
		model = "voyage-3"
	}

	body := voyageEmbeddingRequest{
		Model: model,
		Input: texts,
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.voyageai.com/v1/embeddings"
	}

	resp := voyageEmbeddingResponse{}
	err := callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("Voyage AI embedding API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("Voyage AI embedding API error: %s", resp.Error.Message)
	}

	// Extract embeddings from response
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	embeddings := make([][]float32, len(resp.Data))
	for _, data := range resp.Data {
		embeddings[data.Index] = data.Embedding
	}

	response := &EmbeddingResponse{
		Embeddings: embeddings,
		Metadata:   Metadata{"model": resp.Model},
	}
	if resp.Usage != nil {
		response.Usage = &Usage{InputTokens: resp.Usage.TotalTokens, TotalTokens: resp.Usage.TotalTokens}
	}

	return response, nil
}

// reRank implements the provider interface for Voyage AI reranking
func (p *VoyageProvider) reRank(ctx context.Context, query string, documents []string, cfg CallConfig) (*RerankResponse, error) {
	// Use provided model or default to rerank-2.5
	model := cfg.Model
	if model == "" {
		model = "rerank-2.5"
	}

	body := voyageRerankRequest{
		Model:     model,
		Query:     query,
		Documents: documents,
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.voyageai.com/v1/rerank"
	}

	resp := voyageRerankResponse{}
	err := callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("Voyage AI rerank API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("Voyage AI rerank API error: %s", resp.Error.Message)
	}

	// Extract scores and reorder them to match the original document order
	// The API returns results sorted by relevance, but we need to return scores
	// in the same order as the input documents
	scores := make([]float32, len(documents))
	for _, result := range resp.Results {
		if result.Index >= 0 && result.Index < len(scores) {
			scores[result.Index] = result.RelevanceScore
		}
	}

	response := &RerankResponse{
		Scores:   scores,
		Usage:    &Usage{InputTokens: resp.Usage.TotalTokens, TotalTokens: resp.Usage.TotalTokens},
		Metadata: Metadata{"model": resp.Model},
	}

	return response, nil
}
