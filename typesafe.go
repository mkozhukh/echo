package echo

import (
	"context"
	"fmt"
	"net/http"
	"strings"
)

const typesafeDefaultURL = "https://api.typesafe.ai/v1/systemone"
const typesafeDefaultModel = "jev-latest"

var errTypeSafeOnlyDecisions = fmt.Errorf("TypeSafe only supports decision calls. Use Decide() instead")

// TypeSafeProvider is a stateless provider for TypeSafe System One models (Jev).
// It supports only decision calls (Decide).
type TypeSafeProvider struct {
	Key string
}

// NewTypeSafeClient creates a new TypeSafe client
func NewTypeSafeClient(apiKey, model string, opts ...CallOption) Client {
	if model != "" {
		if !strings.HasPrefix(model, "typesafe/") {
			model = "typesafe/" + model
		}
		opts = append([]CallOption{WithModel(model)}, opts...)
	}
	client, _ := NewClient(opts...)
	client.SetProvider("typesafe", &TypeSafeProvider{Key: apiKey})
	return client
}

type typesafeResponse struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
}

// decide implements the decider interface
func (p *TypeSafeProvider) decide(ctx context.Context, req decisionRequest, cfg CallConfig) (*DecisionResponse, error) {
	if req.Model == "" {
		req.Model = typesafeDefaultModel
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = typesafeDefaultURL
	}

	var resp typesafeResponse
	err := callHTTPAPI(ctx, baseURL, func(r *http.Request) {
		r.Header.Set("Authorization", "Bearer "+p.Key)
	}, req, &resp)
	if err != nil {
		return nil, fmt.Errorf("TypeSafe API call failed: %w", err)
	}

	for id := range req.Questions {
		if _, ok := resp.Answers[id]; !ok {
			return nil, fmt.Errorf("TypeSafe response has no answer for question %q", id)
		}
	}

	out := &DecisionResponse{
		Model:    resp.Model,
		Answers:  resp.Answers,
		Metadata: Metadata{"model": resp.Model},
	}
	if resp.Usage != nil {
		out.Metadata["input_tokens"] = resp.Usage.InputTokens
		out.Metadata["output_tokens"] = resp.Usage.OutputTokens
		out.Metadata["total_tokens"] = resp.Usage.InputTokens + resp.Usage.OutputTokens
	}
	return out, nil
}

// Unsupported provider operations

func (p *TypeSafeProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
	return nil, errTypeSafeOnlyDecisions
}

func (p *TypeSafeProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	return nil, errTypeSafeOnlyDecisions
}
