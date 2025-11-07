package echo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// voyageProvider is a stateless provider for Voyage AI embeddings
// Voyage AI is Anthropic's recommended embedding provider
type voyageProvider struct{}

// Voyage AI structures
type VoyageEmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type VoyageError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type VoyageEmbeddingResponse struct {
	Error *VoyageError `json:"error,omitempty"`
	Data  []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage *struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

type VoyageRerankRequest struct {
	Query      string   `json:"query"`
	Documents  []string `json:"documents"`
	Model      string   `json:"model"`
	TopK       *int     `json:"top_k,omitempty"`
	Truncation *bool    `json:"truncation,omitempty"`
}

type VoyageRerankResponse struct {
	Error   *VoyageError `json:"error,omitempty"`
	Results []struct {
		Index          int     `json:"index"`
		Document       string  `json:"document"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
	TotalTokens int    `json:"total_tokens"`
	Model       string `json:"model"`
}

// call implements the provider interface but returns an error
// Voyage AI only supports embeddings, not chat completions
func (p *voyageProvider) call(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*Response, error) {
	return nil, fmt.Errorf("Voyage AI only supports embeddings, not chat completions. Use GetEmbeddings() instead")
}

// streamCall implements the provider interface but returns an error
// Voyage AI only supports embeddings, not chat completions
func (p *voyageProvider) streamCall(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	return nil, fmt.Errorf("Voyage AI only supports embeddings, not chat completions. Use GetEmbeddings() instead")
}

// getEmbeddings implements the provider interface for Voyage AI embeddings
func (p *voyageProvider) getEmbeddings(ctx context.Context, apiKey string, text string, cfg CallConfig) (*EmbeddingResponse, error) {
	// Use provided model or default to voyage-3
	model := cfg.Model
	if model == "" {
		model = "voyage-3"
	}

	body := VoyageEmbeddingRequest{
		Model: model,
		Input: text,
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.voyageai.com/v1/embeddings"
	}

	resp := VoyageEmbeddingResponse{}
	err := callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("Voyage AI embedding API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("Voyage AI embedding API error: %s", resp.Error.Message)
	}

	// Extract embedding from response
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	response := &EmbeddingResponse{
		Embedding: resp.Data[0].Embedding,
	}

	// Add metadata if usage information is available
	if resp.Usage != nil {
		response.Metadata = Metadata{
			"total_tokens": resp.Usage.TotalTokens,
			"model":        resp.Model,
		}
	}

	return response, nil
}

// reRank implements the provider interface for Voyage AI reranking
func (p *voyageProvider) reRank(ctx context.Context, apiKey string, query string, documents []string, cfg CallConfig) (*RerankResponse, error) {
	// Use provided model or default to rerank-2.5
	model := cfg.Model
	if model == "" {
		model = "rerank-2.5"
	}

	body := VoyageRerankRequest{
		Model:     model,
		Query:     query,
		Documents: documents,
	}

	// Set default base URL if not provided
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.voyageai.com/v1/rerank"
	}

	resp := VoyageRerankResponse{}
	err := callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+apiKey)
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
	scores := make([]float64, len(documents))
	for _, result := range resp.Results {
		if result.Index >= 0 && result.Index < len(scores) {
			scores[result.Index] = result.RelevanceScore
		}
	}

	response := &RerankResponse{
		Scores: scores,
		Metadata: Metadata{
			"total_tokens": resp.TotalTokens,
			"model":        resp.Model,
		},
	}

	return response, nil
}

// parseCompletionRequest parses an HTTP request into a CompletionRequest
// Voyage AI only supports embeddings and reranking, not chat completions
func (p *voyageProvider) parseCompletionRequest(req *http.Request) (*CompletionRequest, error) {
	return nil, fmt.Errorf("Voyage AI only supports embeddings and reranking, not chat completions")
}

// parseEmbeddingRequest parses an HTTP request into an EmbeddingRequest
// Converts from Voyage AI format to OpenAI-compatible format
func (p *voyageProvider) parseEmbeddingRequest(req *http.Request) (*EmbeddingRequest, error) {
	var voyageReq VoyageEmbeddingRequest
	if err := json.NewDecoder(req.Body).Decode(&voyageReq); err != nil {
		return nil, fmt.Errorf("failed to parse Voyage embedding request: %w", err)
	}

	embeddingReq := &EmbeddingRequest{
		Model: voyageReq.Model,
		Input: voyageReq.Input,
	}

	return embeddingReq, nil
}

// parseRerankRequest parses an HTTP request into a RerankRequest
// For Voyage AI, this is a direct JSON parse since we use Voyage format for RerankRequest
func (p *voyageProvider) parseRerankRequest(req *http.Request) (*RerankRequest, error) {
	var rerankReq RerankRequest
	if err := json.NewDecoder(req.Body).Decode(&rerankReq); err != nil {
		return nil, fmt.Errorf("failed to parse Voyage rerank request: %w", err)
	}

	return &rerankReq, nil
}
