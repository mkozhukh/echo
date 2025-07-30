package echo

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type GeminiClient struct {
	apiKey string
	cfg    *CallConfig
}

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

type GeminiResponse struct {
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

// prepareRequest builds the Gemini request with the given configuration
func (c *GeminiClient) prepareRequest(prompt string, opts ...CallOption) (GeminiRequest, CallConfig) {
	// Start with client's default call config
	callCfg := *c.cfg
	// Apply call-specific options (these override client defaults)
	for _, opt := range opts {
		opt(&callCfg)
	}

	// Create Gemini-specific request
	geminiReq := GeminiRequest{
		Contents: []GeminiContent{
			{
				Role: "user",
				Parts: []GeminiPart{
					{Text: prompt},
				},
			},
		},
	}

	// Add system instruction if provided
	if callCfg.SystemMsg != "" {
		geminiReq.SystemInstruction = &GeminiContent{
			Parts: []GeminiPart{
				{Text: callCfg.SystemMsg},
			},
		}
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

	return geminiReq, callCfg
}

func (c *GeminiClient) Call(ctx context.Context, prompt string, opts ...CallOption) (*Response, error) {
	geminiReq, callCfg := c.prepareRequest(prompt, opts...)

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

func (c *GeminiClient) StreamCall(ctx context.Context, prompt string, opts ...CallOption) (*StreamResponse, error) {
	geminiReq, callCfg := c.prepareRequest(prompt, opts...)

	// Update URL for streaming endpoint
	streamURL := strings.Replace(callCfg.BaseURL, ":generateContent", ":streamGenerateContent?alt=sse", 1)

	// Get streaming response
	respBody, err := streamHTTPAPI(ctx, streamURL, func(req *http.Request) {
		req.Header.Set("x-goog-api-key", c.apiKey)
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
			c.processGeminiSSEMessage(msg, ch)
			return nil
		})

		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("SSE stream error: %w", err)}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}

// processGeminiSSEMessage processes individual Gemini SSE messages
func (c *GeminiClient) processGeminiSSEMessage(msg SSEMessage, ch chan StreamChunk) {
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
				Data: []byte(text),
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
