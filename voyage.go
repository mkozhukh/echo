package echo

import (
	"context"
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
