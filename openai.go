package echo

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type openRouterProvider struct {
	Order          []string `json:"order"`
	Only           []string `json:"only"`
	AllowFallbacks bool     `json:"allow_fallbacks"`
}

type openAIError struct {
	Message string `json:"message"`
	Code    int32  `json:"code"`
}

// openAIUsage is the usage block of OpenAI-compatible APIs (OpenAI, OpenRouter, xAI)
type openAIUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"` // OpenRouter
	} `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details,omitempty"`
	Cost *float64 `json:"cost,omitempty"` // OpenRouter: amount charged
}

func (u *openAIUsage) toUsage() *Usage {
	if u == nil {
		return nil
	}
	out := &Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
		CostUSD:      u.Cost,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	if d := u.PromptTokensDetails; d != nil {
		out.CachedInputTokens = d.CachedTokens
		out.CacheWriteTokens = d.CacheWriteTokens
	}
	if d := u.CompletionTokensDetails; d != nil {
		out.ReasoningTokens = d.ReasoningTokens
	}
	return out
}

type openAIRequest struct {
	Model         string          `json:"model"`
	Temperature   *float32        `json:"temperature,omitempty"`
	MaxTokens     *int            `json:"max_completion_tokens,omitempty"`
	Messages      []openAIMessage `json:"messages"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
	Provider        *openRouterProvider   `json:"provider,omitempty"`
	ResponseFormat  *openAIResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort string                `json:"reasoning_effort,omitempty"`
}

// openAIResponseFormat specifies the format for model output
type openAIResponseFormat struct {
	Type       string                  `json:"type"`
	JSONSchema *openAIJSONSchemaConfig `json:"json_schema,omitempty"`
}

// openAIJSONSchemaConfig specifies JSON schema for structured output
type openAIJSONSchemaConfig struct {
	Name   string `json:"name"`
	Strict bool   `json:"strict,omitempty"`
	Schema any    `json:"schema"`
}

// openAIMessage represents a message in OpenAI format
type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIResponse struct {
	Error   *openAIError `json:"error,omitempty"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *openAIUsage `json:"usage,omitempty"`
}

// OpenAIProvider is a stateless provider for OpenAI API
type OpenAIProvider struct {
	Key string
	// Host optionally overrides the origin used when cfg.BaseURL is empty,
	// so OpenAI-compatible servers (llama.cpp, Ollama, vLLM, LM Studio, etc.)
	// can be reached without a per-endpoint URL. Leave empty to hit
	// api.openai.com. Endpoint paths (/v1/chat/completions, /v1/embeddings)
	// are appended automatically.
	Host string
}

// resolveURL picks the request URL for the given endpoint path. An explicit
// cfg.BaseURL (full URL including path) wins; otherwise p.Host is used as an
// origin; otherwise we fall back to OpenAI's public endpoint.
func (p *OpenAIProvider) resolveURL(cfg CallConfig, endpointPath string) string {
	if cfg.BaseURL != "" {
		return cfg.BaseURL
	}
	host := p.Host
	if host == "" {
		host = "https://api.openai.com"
	}
	return strings.TrimRight(host, "/") + endpointPath
}

// NewOpenAIClient creates a new OpenAI client (deprecated, kept for compatibility)
func NewOpenAIClient(apiKey, model string, opts ...CallOption) Client {
	if model != "" {
		if !strings.HasPrefix(model, "openai/") {
			model = "openai/" + model
		}
		opts = append([]CallOption{WithModel(model)}, opts...)
	}
	client, _ := NewClient(opts...)
	client.SetProvider("openai", &OpenAIProvider{Key: apiKey})
	return client
}

// prepareOpenAIRequest builds the OpenAI request with the given configuration
func prepareOpenAIRequest(messages []Message, streaming bool, cfg CallConfig) (openAIRequest, error) {
	// Validate messages
	if err := validateMessages(messages); err != nil {
		return openAIRequest{}, fmt.Errorf("invalid message chain: %w", err)
	}

	// Convert messages to OpenAI format
	openaiMessages := []openAIMessage{}
	systemMessageProcessed := false

	for _, msg := range messages {
		switch msg.Role {
		case System:
			// Skip system message here if WithSystemMessage is set
			if cfg.SystemMsg == "" {
				openaiMessages = append(openaiMessages, openAIMessage{
					Role:    "system",
					Content: msg.Content,
				})
			}
			systemMessageProcessed = true
		case User:
			openaiMessages = append(openaiMessages, openAIMessage{
				Role:    "user",
				Content: msg.Content,
			})
		case Agent:
			openaiMessages = append(openaiMessages, openAIMessage{
				Role:    "assistant",
				Content: msg.Content,
			})
		}
	}

	// Handle WithSystemMessage option
	if cfg.SystemMsg != "" {
		// Insert system message at the beginning
		systemMsg := openAIMessage{
			Role:    "system",
			Content: cfg.SystemMsg,
		}
		if systemMessageProcessed {
			// Replace the first message (which should be system)
			openaiMessages = append([]openAIMessage{systemMsg}, openaiMessages[1:]...)
		} else {
			// Add system message at the beginning
			openaiMessages = append([]openAIMessage{systemMsg}, openaiMessages...)
		}
	}

	req := openAIRequest{
		Model:       cfg.Model,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		Messages:    openaiMessages,
		Stream:      streaming,
	}

	// Add stream options for usage stats when streaming
	if streaming {
		req.StreamOptions = &struct {
			IncludeUsage bool `json:"include_usage"`
		}{
			IncludeUsage: true,
		}
	}

	// Add provider field if EndPoint is set (for openrouter compatibility)
	if cfg.EndPoint != "" {
		order := strings.Split(cfg.EndPoint, ",")
		req.Provider = &openRouterProvider{
			Only:           order,
			Order:          order,
			AllowFallbacks: true,
		}
	}

	// Add structured output response format if configured
	if cfg.StructuredOutput != nil {
		req.ResponseFormat = &openAIResponseFormat{
			Type: "json_schema",
			JSONSchema: &openAIJSONSchemaConfig{
				Name:   cfg.StructuredOutput.Name,
				Strict: true,
				Schema: cfg.StructuredOutput.Schema,
			},
		}
	}

	// Add reasoning effort if configured (for o1 models)
	if cfg.ReasoningEffort != "" {
		req.ReasoningEffort = cfg.ReasoningEffort
	}

	return req, nil
}

// call implements the provider interface for OpenAI
func (p *OpenAIProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
	body, err := prepareOpenAIRequest(messages, false, cfg)
	if err != nil {
		return nil, err
	}

	baseURL := p.resolveURL(cfg, "/v1/chat/completions")

	resp := openAIResponse{}
	err = callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("OpenAI API error: %s", resp.Error.Message)
	}

	// Extract text from LLM response
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	return &Response{
		Text:  resp.Choices[0].Message.Content,
		Usage: resp.Usage.toUsage(),
	}, nil
}

// Streaming response structures
type openAIStreamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *openAIUsage `json:"usage,omitempty"`
}

// streamCall implements the provider interface for OpenAI streaming
func (p *OpenAIProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	body, err := prepareOpenAIRequest(messages, true, cfg)
	if err != nil {
		return nil, err
	}

	baseURL := p.resolveURL(cfg, "/v1/chat/completions")

	// Get streaming response
	respBody, err := streamHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}, body)
	if err != nil {
		return nil, fmt.Errorf("OpenAI streaming API call failed: %w", err)
	}

	// Create channel for streaming
	ch := make(chan StreamChunk)

	// Start goroutine to process stream
	go func() {
		defer close(ch)
		defer respBody.Close()

		reader := bufio.NewReader(respBody)
		for {
			line, err := reader.ReadBytes('\n')
			if err == io.EOF {
				break
			}
			if err != nil {
				ch <- StreamChunk{Error: fmt.Errorf("read error: %w", err)}
				return
			}

			// Skip empty lines
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			// Check for SSE data prefix
			if !bytes.HasPrefix(line, dataPrefix) {
				continue
			}

			// Remove "data: " prefix
			data := bytes.TrimPrefix(line, dataPrefix)

			// Check for end of stream
			if bytes.Equal(data, doneMarker) {
				return
			}

			// Parse JSON
			var streamResp openAIStreamResponse
			if err := json.Unmarshal(data, &streamResp); err != nil {
				ch <- StreamChunk{Error: fmt.Errorf("json parse error: %w", err)}
				return
			}

			if len(streamResp.Choices) > 0 && streamResp.Choices[0].Delta.Content != "" {
				ch <- StreamChunk{
					Data: streamResp.Choices[0].Delta.Content,
				}
			}

			// Usage arrives on the final chunk (some servers also include choices there)
			if streamResp.Usage != nil {
				ch <- StreamChunk{
					Usage: streamResp.Usage.toUsage(),
				}
			}
		}
	}()

	return &StreamResponse{Stream: ch}, nil
}

// OpenAI Embedding structures
type openAIEmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbeddingResponse struct {
	Error *openAIError `json:"error,omitempty"`
	Data  []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage *openAIUsage `json:"usage,omitempty"`
}

// getEmbeddings implements the provider interface for OpenAI embeddings
func (p *OpenAIProvider) getEmbeddings(ctx context.Context, texts []string, cfg CallConfig) (*EmbeddingResponse, error) {
	// Use provided model or default to text-embedding-3-small
	model := cfg.Model
	if model == "" {
		model = "text-embedding-3-small"
	}

	body := openAIEmbeddingRequest{
		Model: model,
		Input: texts,
	}

	baseURL := p.resolveURL(cfg, "/v1/embeddings")

	resp := openAIEmbeddingResponse{}
	err := callHTTPAPI(ctx, baseURL, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}, body, &resp)
	if err != nil {
		return nil, fmt.Errorf("OpenAI embedding API call failed: %w", err)
	}

	// Check for errors in the response
	if resp.Error != nil {
		return nil, fmt.Errorf("OpenAI embedding API error: %s", resp.Error.Message)
	}

	// Extract embeddings from response
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	embeddings := make([][]float32, len(resp.Data))
	for _, data := range resp.Data {
		embeddings[data.Index] = data.Embedding
	}

	return &EmbeddingResponse{
		Embeddings: embeddings,
		Usage:      resp.Usage.toUsage(),
	}, nil
}
