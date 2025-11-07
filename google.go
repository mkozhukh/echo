package echo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// googleProvider is a stateless provider for Google API
type googleProvider struct{}

// Gemini-specific request/response structures
type GeminiRequest struct {
	Contents          []GeminiContent `json:"contents"`
	SystemInstruction *GeminiContent  `json:"systemInstruction,omitempty"`
	GenerationConfig  *struct {
		Temperature     *float64 `json:"temperature,omitempty"`
		MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	} `json:"generationConfig,omitempty"`
}

type GeminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text string `json:"text"`
}

type GeminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

type GeminiResponse struct {
	Error      *GeminiError `json:"error,omitempty"`
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata,omitempty"`
}

// GeminiStreamResponse represents a streaming response chunk from Gemini
type GeminiStreamResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata,omitempty"`
}

// NewGoogleClient creates a new Google client (deprecated, kept for compatibility)
func NewGoogleClient(apiKey, model string, opts ...CallOption) *CommonClient {
	client, _ := NewCommonClient("google/"+model, apiKey, opts...)
	return client
}

// prepareGoogleRequest builds the Gemini request with the given configuration
func prepareGoogleRequest(messages []Message, cfg CallConfig) (GeminiRequest, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return GeminiRequest{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Convert messages to Gemini format
	geminiContents := []GeminiContent{}
	var systemMsg string

	for _, msg := range messages {
		switch msg.Role {
		case System:
			systemMsg = msg.Content
		case User:
			geminiContents = append(geminiContents, GeminiContent{
				Role: "user",
				Parts: []GeminiPart{
					{Text: msg.Content},
				},
			})
		case Agent:
			geminiContents = append(geminiContents, GeminiContent{
				Role: "model",
				Parts: []GeminiPart{
					{Text: msg.Content},
				},
			})
		}
	}

	// Create Gemini-specific request
	geminiReq := GeminiRequest{
		Contents: geminiContents,
	}

	// Handle system instruction - WithSystemMessage overrides message chain system
	if cfg.SystemMsg != "" {
		geminiReq.SystemInstruction = &GeminiContent{
			Parts: []GeminiPart{
				{Text: cfg.SystemMsg},
			},
		}
	} else if systemMsg != "" {
		geminiReq.SystemInstruction = &GeminiContent{
			Parts: []GeminiPart{
				{Text: systemMsg},
			},
		}
	}

	// Add generation config if temperature or max tokens are set
	if cfg.Temperature != nil || cfg.MaxTokens != nil {
		geminiReq.GenerationConfig = &struct {
			Temperature     *float64 `json:"temperature,omitempty"`
			MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
		}{
			Temperature:     cfg.Temperature,
			MaxOutputTokens: cfg.MaxTokens,
		}
	}

	return geminiReq, nil
}

// call implements the provider interface for Google
func (p *googleProvider) call(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*Response, error) {
	geminiReq, err := prepareGoogleRequest(messages, cfg)
	if err != nil {
		return nil, err
	}

	// Build the base URL with model
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta/models/" + cfg.Model + ":generateContent"
	}

	// Call the Gemini API using shared HTTP function
	var response GeminiResponse
	err = callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("x-goog-api-key", apiKey)
	}, geminiReq, &response)
	if err != nil {
		return nil, fmt.Errorf("api call failed: %w", err)
	}

	// Check for errors in the response
	if response.Error != nil {
		return nil, fmt.Errorf("Gemini API error: %s", response.Error.Message)
	}

	if len(response.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in Gemini response")
	}

	if len(response.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no content parts in Gemini response")
	}

	result := &Response{Text: response.Candidates[0].Content.Parts[0].Text}

	// Add metadata if usage information is available
	if response.UsageMetadata != nil {
		result.Metadata = Metadata{
			"total_tokens":      response.UsageMetadata.TotalTokenCount,
			"prompt_tokens":     response.UsageMetadata.PromptTokenCount,
			"completion_tokens": response.UsageMetadata.CandidatesTokenCount,
		}
	}

	return result, nil
}

// streamCall implements the provider interface for Google streaming
func (p *googleProvider) streamCall(ctx context.Context, apiKey string, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	geminiReq, err := prepareGoogleRequest(messages, cfg)
	if err != nil {
		return nil, err
	}

	// Build the base URL with model
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta/models/" + cfg.Model + ":generateContent"
	}

	// Update URL for streaming endpoint
	streamURL := strings.Replace(baseURL, ":generateContent", ":streamGenerateContent?alt=sse", 1)

	// Get streaming response
	respBody, err := streamHTTPAPI(ctx, streamURL, func(req *http.Request) {
		req.Header.Set("x-goog-api-key", apiKey)
	}, geminiReq)
	if err != nil {
		return nil, fmt.Errorf("Gemini streaming API call failed: %w", err)
	}

	// Create channel for streaming
	ch := make(chan StreamChunk)

	// Start goroutine to process stream
	go func() {
		defer close(ch)

		err := parseSSEStream(respBody, func(msg SSEMessage) error {
			processGeminiSSEMessage(msg, ch)
			return nil
		})

		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("SSE stream error: %w", err)}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}

// processGeminiSSEMessage processes individual Gemini SSE messages
func processGeminiSSEMessage(msg SSEMessage, ch chan StreamChunk) {
	if len(msg.Data) == 0 {
		return
	}

	// Parse JSON
	var streamResp GeminiStreamResponse
	if err := json.Unmarshal(msg.Data, &streamResp); err != nil {
		ch <- StreamChunk{Error: fmt.Errorf("json parse error: %w", err)}
		return
	}

	// Check if we have candidates with content
	if len(streamResp.Candidates) > 0 && len(streamResp.Candidates[0].Content.Parts) > 0 {
		text := streamResp.Candidates[0].Content.Parts[0].Text
		if text != "" {
			ch <- StreamChunk{
				Data: text,
			}
		}
	}

	// Check if this chunk contains usage metadata
	if streamResp.UsageMetadata != nil {
		meta := Metadata{
			"total_tokens":      streamResp.UsageMetadata.TotalTokenCount,
			"prompt_tokens":     streamResp.UsageMetadata.PromptTokenCount,
			"completion_tokens": streamResp.UsageMetadata.CandidatesTokenCount,
		}
		ch <- StreamChunk{
			Meta: &meta,
		}
	}
}

// Google Embedding structures
type GoogleEmbeddingRequest struct {
	Content GeminiContent `json:"content"`
}

type GoogleEmbeddingResponse struct {
	Error     *GeminiError `json:"error,omitempty"`
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
}

// getEmbeddings implements the provider interface for Google embeddings
func (p *googleProvider) getEmbeddings(ctx context.Context, apiKey string, text string, cfg CallConfig) (*EmbeddingResponse, error) {
	// Use provided model or default to text-embedding-004
	model := cfg.Model
	if model == "" {
		model = "text-embedding-004"
	}

	body := GoogleEmbeddingRequest{
		Content: GeminiContent{
			Parts: []GeminiPart{
				{Text: text},
			},
		},
	}

	// Build the base URL with model
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":embedContent"
	}

	resp := GoogleEmbeddingResponse{}
	err := callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("x-goog-api-key", apiKey)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("Google embedding API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("Google embedding API error: %s", resp.Error.Message)
	}

	// Extract embedding from response
	if len(resp.Embedding.Values) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	response := &EmbeddingResponse{
		Embedding: resp.Embedding.Values,
		Metadata:  Metadata{},
	}

	return response, nil
}
