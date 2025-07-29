package echo

import (
	"context"
	"fmt"
	"net/http"
)

type GeminiClient struct {
	apiKey string
	cfg    *CallConfig
}

// Gemini-specific request/response structures
type GeminiRequest struct {
	Contents []struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"contents"`
	GenerationConfig *struct {
		Temperature     *float64 `json:"temperature,omitempty"`
		MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	} `json:"generationConfig,omitempty"`
}

type GeminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// NewGeminiClient creates a new Gemini client with full configuration
func NewGeminiClient(apiKey, model string, opts ...CallOption) *GeminiClient {
	cfg := &CallConfig{
		Model:   model,
		BaseURL: "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent",
	}

	// Apply client options
	for _, opt := range opts {
		opt(cfg)
	}

	return &GeminiClient{apiKey: apiKey, cfg: cfg}
}

func NewGemini25Pro(apiKey string, opts ...CallOption) *GeminiClient {
	return NewGeminiClient(apiKey, "gemini-2.5-pro", opts...)
}

func NewGemini25Flash(apiKey string, opts ...CallOption) *GeminiClient {
	return NewGeminiClient(apiKey, "gemini-2.5-flash", opts...)
}

func (c *GeminiClient) Call(ctx context.Context, prompt string, opts ...CallOption) (*Response, error) {
	// Start with client's default call config
	callCfg := *c.cfg
	// Apply call-specific options (these override client defaults)
	for _, opt := range opts {
		opt(&callCfg)
	}

	// Create Gemini-specific request
	geminiReq := GeminiRequest{
		Contents: []struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		}{
			{
				Parts: []struct {
					Text string `json:"text"`
				}{
					{Text: prompt},
				},
			},
		},
	}

	// Add generation config if temperature or max tokens are set
	if callCfg.Temperature != nil || callCfg.MaxTokens != nil {
		geminiReq.GenerationConfig = &struct {
			Temperature     *float64 `json:"temperature,omitempty"`
			MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
		}{
			Temperature:     callCfg.Temperature,
			MaxOutputTokens: callCfg.MaxTokens,
		}
	}

	// Call the Gemini API using shared HTTP function
	var response GeminiResponse
	err := callHTTPAPI(ctx, callCfg.BaseURL, func(req *http.Request) {
		req.Header.Set("x-goog-api-key", c.apiKey)
	}, geminiReq, &response)
	if err != nil {
		return nil, fmt.Errorf("api call failed: %w", err)
	}

	if len(response.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in Gemini response")
	}

	if len(response.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no content parts in Gemini response")
	}

	return &Response{Text: response.Candidates[0].Content.Parts[0].Text}, nil
}
