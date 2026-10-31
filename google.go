package echo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// GoogleProvider is a stateless provider for Google API
type GoogleProvider struct {
	Key string
}

// Gemini-specific request/response structures
type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

// geminiGenerationConfig contains generation parameters for Gemini requests
type geminiGenerationConfig struct {
	Temperature      *float32              `json:"temperature,omitempty"`
	MaxOutputTokens  *int                  `json:"maxOutputTokens,omitempty"`
	ResponseMimeType string                `json:"responseMimeType,omitempty"`
	ResponseSchema   any                   `json:"responseSchema,omitempty"`
	ThinkingConfig   *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

// geminiThinkingConfig contains thinking/reasoning configuration
type geminiThinkingConfig struct {
	ThinkingLevel string `json:"thinkingLevel"` // "low", "medium", "high"
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// geminiUsage is the usageMetadata block of Gemini responses.
// promptTokenCount includes cached tokens; thinking tokens are reported
// separately from candidatesTokenCount but billed as output.
type geminiUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
}

func (u *geminiUsage) toUsage() *Usage {
	if u == nil {
		return nil
	}
	out := &Usage{
		InputTokens:       u.PromptTokenCount,
		CachedInputTokens: u.CachedContentTokenCount,
		OutputTokens:      u.CandidatesTokenCount + u.ThoughtsTokenCount,
		ReasoningTokens:   u.ThoughtsTokenCount,
		TotalTokens:       u.TotalTokenCount,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	return out
}

type geminiResponse struct {
	Error      *geminiError `json:"error,omitempty"`
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata *geminiUsage `json:"usageMetadata,omitempty"`
}

// geminiStreamResponse represents a streaming response chunk from Gemini
type geminiStreamResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata *geminiUsage `json:"usageMetadata,omitempty"`
}

// NewGoogleClient creates a new Google client (deprecated, kept for compatibility)
func NewGoogleClient(apiKey, model string, opts ...CallOption) Client {
	if model != "" {
		if !strings.HasPrefix(model, "google/") {
			model = "google/" + model
		}
		opts = append([]CallOption{WithModel(model)}, opts...)
	}
	client, _ := NewClient(opts...)
	client.SetProvider("google", &GoogleProvider{Key: apiKey})
	return client
}

// prepareGoogleRequest builds the Gemini request with the given configuration
func prepareGoogleRequest(messages []Message, cfg CallConfig) (geminiRequest, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return geminiRequest{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Convert messages to Gemini format
	geminiContents := []geminiContent{}
	var systemMsg string

	for _, msg := range messages {
		switch msg.Role {
		case System:
			systemMsg = msg.Content
		case User:
			geminiContents = append(geminiContents, geminiContent{
				Role: "user",
				Parts: []geminiPart{
					{Text: msg.Content},
				},
			})
		case Agent:
			geminiContents = append(geminiContents, geminiContent{
				Role: "model",
				Parts: []geminiPart{
					{Text: msg.Content},
				},
			})
		}
	}

	// Create Gemini-specific request
	geminiReq := geminiRequest{
		Contents: geminiContents,
	}

	// Handle system instruction - WithSystemMessage overrides message chain system
	if cfg.SystemMsg != "" {
		geminiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{
				{Text: cfg.SystemMsg},
			},
		}
	} else if systemMsg != "" {
		geminiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{
				{Text: systemMsg},
			},
		}
	}

	// Add generation config if temperature, max tokens, structured output, or reasoning effort are set
	if cfg.Temperature != nil || cfg.MaxTokens != nil || cfg.StructuredOutput != nil || cfg.ReasoningEffort != "" {
		geminiReq.GenerationConfig = &geminiGenerationConfig{
			Temperature:     cfg.Temperature,
			MaxOutputTokens: cfg.MaxTokens,
		}

		// Add structured output configuration
		if cfg.StructuredOutput != nil {
			geminiReq.GenerationConfig.ResponseMimeType = "application/json"
			geminiReq.GenerationConfig.ResponseSchema = cfg.StructuredOutput.Schema
		}

		// Add thinking/reasoning configuration
		if cfg.ReasoningEffort != "" {
			geminiReq.GenerationConfig.ThinkingConfig = &geminiThinkingConfig{
				ThinkingLevel: cfg.ReasoningEffort,
			}
		}
	}

	return geminiReq, nil
}

// call implements the provider interface for Google
func (p *GoogleProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
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
	var response geminiResponse
	err = callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("x-goog-api-key", p.Key)
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

	return &Response{
		Text:  response.Candidates[0].Content.Parts[0].Text,
		Usage: response.UsageMetadata.toUsage(),
	}, nil
}

// streamCall implements the provider interface for Google streaming
func (p *GoogleProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
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
		req.Header.Set("x-goog-api-key", p.Key)
	}, geminiReq)
	if err != nil {
		return nil, fmt.Errorf("Gemini streaming API call failed: %w", err)
	}

	// Create channel for streaming
	ch := make(chan StreamChunk)

	// Start goroutine to process stream
	go func() {
		defer close(ch)

		// usageMetadata is cumulative and repeated on chunks; report the last one
		var usage *geminiUsage
		err := parseSSEStream(respBody, func(msg sseMessage) error {
			if u := processGeminiSSEMessage(msg, ch); u != nil {
				usage = u
			}
			return nil
		})

		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("SSE stream error: %w", err)}
			return
		}
		if usage != nil {
			ch <- StreamChunk{Usage: usage.toUsage()}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}

// processGeminiSSEMessage processes individual Gemini SSE messages and
// returns the usage block of the chunk, if present
func processGeminiSSEMessage(msg sseMessage, ch chan StreamChunk) *geminiUsage {
	if len(msg.Data) == 0 {
		return nil
	}

	// Parse JSON
	var streamResp geminiStreamResponse
	if err := json.Unmarshal(msg.Data, &streamResp); err != nil {
		ch <- StreamChunk{Error: fmt.Errorf("json parse error: %w", err)}
		return nil
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

	return streamResp.UsageMetadata
}

type googleBatchEmbeddingRequest struct {
	Requests []googleBatchEmbeddingRequestItem `json:"requests"`
}

type googleBatchEmbeddingRequestItem struct {
	Model   string        `json:"model"`
	Content geminiContent `json:"content"`
}

type googleBatchEmbeddingResponse struct {
	Error      *geminiError `json:"error,omitempty"`
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}

// getEmbeddings implements the provider interface for Google embeddings
func (p *GoogleProvider) getEmbeddings(ctx context.Context, texts []string, cfg CallConfig) (*EmbeddingResponse, error) {
	// Use provided model or default to gemini-embedding-2
	model := cfg.Model
	if model == "" {
		model = "gemini-embedding-2"
	}

	// Build batch request
	requests := make([]googleBatchEmbeddingRequestItem, len(texts))
	for i, text := range texts {
		requests[i] = googleBatchEmbeddingRequestItem{
			Model: "models/" + model,
			Content: geminiContent{
				Parts: []geminiPart{{Text: text}},
			},
		}
	}

	body := googleBatchEmbeddingRequest{Requests: requests}

	// Build the base URL with model
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":batchEmbedContents"
	}

	resp := googleBatchEmbeddingResponse{}
	err := callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("x-goog-api-key", p.Key)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("Google embedding API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("Google embedding API error: %s", resp.Error.Message)
	}

	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	embeddings := make([][]float32, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		embeddings[i] = emb.Values
	}

	return &EmbeddingResponse{
		Embeddings: embeddings,
		Metadata:   Metadata{},
	}, nil
}
