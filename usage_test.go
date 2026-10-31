package echo

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func serve(t *testing.T, contentType, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func streamUsage(t *testing.T, resp *StreamResponse) (string, *Usage) {
	t.Helper()
	var text string
	var usage *Usage
	for chunk := range resp.Stream {
		if chunk.Error != nil {
			t.Fatalf("stream error: %v", chunk.Error)
		}
		text += chunk.Data
		if chunk.Usage != nil {
			if usage != nil {
				t.Errorf("usage reported more than once")
			}
			usage = chunk.Usage
		}
	}
	return text, usage
}

func checkUsage(t *testing.T, got *Usage, want Usage) {
	t.Helper()
	if got == nil {
		t.Fatalf("usage is nil, want %+v", want)
	}
	g := *got
	g.CostUSD, want.CostUSD = nil, nil
	if g != want {
		t.Errorf("usage = %+v, want %+v", g, want)
	}
}

func TestUsageAdd(t *testing.T) {
	c := 0.5
	u := Usage{InputTokens: 1, TotalTokens: 1}
	u.Add(nil)
	u.Add(&Usage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5, CostUSD: &c})
	u.Add(&Usage{CostUSD: &c})
	if u.InputTokens != 3 || u.OutputTokens != 3 || u.TotalTokens != 6 || u.CostUSD == nil || *u.CostUSD != 1 {
		t.Errorf("Add = %+v", u)
	}
}

// single line: it is also embedded in SSE data lines
const openAIUsageJSON = `"usage":{"prompt_tokens":100,"completion_tokens":40,"total_tokens":140,` +
	`"prompt_tokens_details":{"cached_tokens":60,"cache_write_tokens":10},` +
	`"completion_tokens_details":{"reasoning_tokens":25},"cost":0.0123}`

var openAIWant = Usage{InputTokens: 100, CachedInputTokens: 60, CacheWriteTokens: 10, OutputTokens: 40, ReasoningTokens: 25, TotalTokens: 140}

func TestOpenAIUsage(t *testing.T) {
	ctx := context.Background()
	srv := serve(t, "application/json", `{"choices":[{"message":{"content":"hi"}}],`+openAIUsageJSON+`}`)
	resp, err := NewOpenAIClient("k", "gpt-5", WithBaseURL(srv.URL)).Complete(ctx, QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	checkUsage(t, resp.Usage, openAIWant)
	if resp.Usage.CostUSD == nil || *resp.Usage.CostUSD != 0.0123 {
		t.Errorf("cost = %v", resp.Usage.CostUSD)
	}
	if resp.Metadata != nil {
		t.Errorf("metadata should not carry tokens: %v", resp.Metadata)
	}

	// usage on the last chunk, with an empty choices list
	srv = serve(t, "text/event-stream", "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n"+
		"data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n"+
		"data: {\"choices\":[],"+openAIUsageJSON+"}\n\ndata: [DONE]\n\n")
	sresp, err := NewOpenAIClient("k", "gpt-5", WithBaseURL(srv.URL)).StreamComplete(ctx, QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	text, usage := streamUsage(t, sresp)
	if text != "hello" {
		t.Errorf("text = %q", text)
	}
	checkUsage(t, usage, openAIWant)

	// embeddings: no cost, no output tokens
	srv = serve(t, "application/json", `{"data":[{"embedding":[1],"index":0}],"usage":{"prompt_tokens":7,"total_tokens":7}}`)
	eresp, err := NewOpenAIClient("k", "text-embedding-3-small", WithBaseURL(srv.URL)).GetEmbeddings(ctx, []string{"a"})
	if err != nil {
		t.Fatal(err)
	}
	checkUsage(t, eresp.Usage, Usage{InputTokens: 7, TotalTokens: 7})
	if eresp.Usage.CostUSD != nil {
		t.Errorf("unexpected cost")
	}
}

func TestXAIUsage(t *testing.T) {
	srv := serve(t, "application/json", `{"choices":[{"message":{"content":"hi"}}],`+openAIUsageJSON+`}`)
	resp, err := NewXAIClient("k", "grok-4", WithBaseURL(srv.URL)).Complete(context.Background(), QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	checkUsage(t, resp.Usage, openAIWant)
}

func TestAnthropicUsage(t *testing.T) {
	ctx := context.Background()
	srv := serve(t, "application/json", `{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":20,"cache_creation_input_tokens":30,"cache_read_input_tokens":40}}`)
	resp, err := NewAnthropicClient("k", "claude-haiku-4-5", WithBaseURL(srv.URL)).Complete(ctx, QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	want := Usage{InputTokens: 80, CachedInputTokens: 40, CacheWriteTokens: 30, OutputTokens: 20, TotalTokens: 100}
	checkUsage(t, resp.Usage, want)
	if resp.Metadata["stop_reason"] != "end_turn" || len(resp.Metadata) != 1 {
		t.Errorf("metadata = %v", resp.Metadata)
	}

	srv = serve(t, "text/event-stream",
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":1,\"cache_creation_input_tokens\":30,\"cache_read_input_tokens\":40}}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"+
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":20}}\n\n"+
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	sresp, err := NewAnthropicClient("k", "claude-haiku-4-5", WithBaseURL(srv.URL)).StreamComplete(ctx, QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	text, usage := streamUsage(t, sresp)
	if text != "hi" {
		t.Errorf("text = %q", text)
	}
	checkUsage(t, usage, want)
}

func TestGoogleUsage(t *testing.T) {
	ctx := context.Background()
	usageJSON := `"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"thoughtsTokenCount":30,"cachedContentTokenCount":50,"totalTokenCount":150}`
	want := Usage{InputTokens: 100, CachedInputTokens: 50, OutputTokens: 50, ReasoningTokens: 30, TotalTokens: 150}

	srv := serve(t, "application/json", `{"candidates":[{"content":{"parts":[{"text":"hi"}]}}],`+usageJSON+`}`)
	resp, err := NewGoogleClient("k", "gemini-2.5-flash", WithBaseURL(srv.URL+"/m:generateContent")).Complete(ctx, QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	checkUsage(t, resp.Usage, want)

	// cumulative usage repeated on each chunk: reported once, last value wins
	srv = serve(t, "text/event-stream",
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"he\"}]}}],\"usageMetadata\":{\"promptTokenCount\":100,\"totalTokenCount\":100}}\n\n"+
			"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"llo\"}]}}],"+usageJSON+"}\n\n")
	sresp, err := NewGoogleClient("k", "gemini-2.5-flash", WithBaseURL(srv.URL+"/m:generateContent")).StreamComplete(ctx, QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	text, usage := streamUsage(t, sresp)
	if text != "hello" {
		t.Errorf("text = %q", text)
	}
	checkUsage(t, usage, want)
}

func TestVoyageUsage(t *testing.T) {
	ctx := context.Background()
	srv := serve(t, "application/json", `{"data":[{"embedding":[1],"index":0}],"model":"voyage-4","usage":{"total_tokens":9}}`)
	eresp, err := NewVoyageClient("k", "voyage-4", WithBaseURL(srv.URL)).GetEmbeddings(ctx, []string{"a"})
	if err != nil {
		t.Fatal(err)
	}
	checkUsage(t, eresp.Usage, Usage{InputTokens: 9, TotalTokens: 9})

	srv = serve(t, "application/json", `{"data":[{"index":0,"relevance_score":0.9}],"model":"rerank-2.5","usage":{"total_tokens":11}}`)
	rresp, err := NewVoyageClient("k", "rerank-2.5", WithBaseURL(srv.URL)).ReRank(ctx, "q", []string{"a"})
	if err != nil {
		t.Fatal(err)
	}
	checkUsage(t, rresp.Usage, Usage{InputTokens: 11, TotalTokens: 11})
}
