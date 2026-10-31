package pricing

import (
	"bytes"
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mkozhukh/echo"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestDefaultTable(t *testing.T) {
	p := Default()
	for _, m := range []string{
		"openai/gpt-5.2", "openai/text-embedding-3-small", "anthropic/claude-haiku-4-5",
		"google/gemini-2.5-pro", "openrouter/openai/gpt-5",
	} {
		if price, ok := p.Get(m); !ok || price.Input <= 0 {
			t.Errorf("%s: price = %+v, ok = %v", m, price, ok)
		}
	}

	// hand-maintained prices
	for m, in := range map[string]float64{
		"voyage/voyage-4-large": 0.12, "voyage/voyage-4": 0.06, "voyage/voyage-4-lite": 0.02,
		"voyage/rerank-2.5": 0.05, "typesafe/jev-latest": 0.04, "typesafe/jev-1.13.0": 0.04,
	} {
		if price, ok := p.Get(m); !ok || price.Input != in {
			t.Errorf("%s: price = %+v, ok = %v, want input %v", m, price, ok, in)
		}
	}

	if _, ok := p.Get("llama/whatever"); ok {
		t.Errorf("unexpected price for local model")
	}
}

func TestCost(t *testing.T) {
	p := Default().Clone()
	p.Set("test/llm", Price{Rates: Rates{Input: 1, Output: 5, CacheRead: 0.1, CacheWrite: 1.25}})

	u := &echo.Usage{Model: "test/llm", InputTokens: 1_000_000, CachedInputTokens: 400_000, CacheWriteTokens: 100_000, OutputTokens: 200_000, ReasoningTokens: 50_000}
	c, ok := p.Cost(u)
	if !ok {
		t.Fatal("no price")
	}
	want := Cost{Input: 0.5, CachedInput: 0.04, CacheWrite: 0.125, Output: 1.0, Total: 1.665}
	if !near(c.Input, want.Input) || !near(c.CachedInput, want.CachedInput) || !near(c.CacheWrite, want.CacheWrite) ||
		!near(c.Output, want.Output) || c.Reasoning != 0 || !near(c.Total, want.Total) || c.Reported {
		t.Errorf("cost = %+v, want %+v", c, want)
	}

	// no cache prices: cached tokens billed as input
	p.Set("test/nocache", Price{Rates: Rates{Input: 2, Output: 4}})
	c, _ = p.Cost(&echo.Usage{Model: "test/nocache", InputTokens: 1_000_000, CachedInputTokens: 500_000})
	if !near(c.Total, 2) {
		t.Errorf("nocache total = %v", c.Total)
	}

	// separately priced reasoning
	p.Set("test/reason", Price{Rates: Rates{Input: 1, Output: 2, Reasoning: 10}})
	c, _ = p.Cost(&echo.Usage{Model: "test/reason", OutputTokens: 1_000_000, ReasoningTokens: 250_000})
	if !near(c.Output, 1.5) || !near(c.Reasoning, 2.5) || !near(c.Total, 4) {
		t.Errorf("reasoning cost = %+v", c)
	}

	// context tiers apply to the whole call
	p.Set("test/tiered", Price{Rates: Rates{Input: 1, Output: 10}, Tiers: []Tier{{Above: 200_000, Rates: Rates{Input: 2, Output: 15}}}})
	c, _ = p.Cost(&echo.Usage{Model: "test/tiered", InputTokens: 200_000, OutputTokens: 1000})
	if !near(c.Total, 0.2+0.01) {
		t.Errorf("below tier total = %v", c.Total)
	}
	c, _ = p.Cost(&echo.Usage{Model: "test/tiered", InputTokens: 300_000, OutputTokens: 1000})
	if !near(c.Total, 0.6+0.015) {
		t.Errorf("above tier total = %v", c.Total)
	}

	// provider-reported cost wins
	reported := 0.42
	c, ok = p.Cost(&echo.Usage{Model: "unknown/model", CostUSD: &reported})
	if !ok || !c.Reported || c.Total != 0.42 {
		t.Errorf("reported cost = %+v %v", c, ok)
	}

	if _, ok := p.Cost(&echo.Usage{Model: "unknown/model"}); ok {
		t.Errorf("unknown model should not be priced")
	}
	if _, ok := p.Cost(nil); ok {
		t.Errorf("nil usage should not be priced")
	}
}

func TestEmbeddedTiersParsed(t *testing.T) {
	price, ok := Default().Get("google/gemini-2.5-pro")
	if !ok || len(price.Tiers) == 0 || price.Tiers[0].Above != 200_000 || price.Tiers[0].Input <= price.Input {
		t.Errorf("gemini-2.5-pro tiers = %+v", price)
	}
}

func TestEnrichAndClone(t *testing.T) {
	p := Default().Clone()
	p.Set("test/llm", Price{Rates: Rates{Input: 1, Output: 1}})
	if _, ok := Default().Get("test/llm"); ok {
		t.Errorf("Set on clone leaked into Default")
	}

	u := &echo.Usage{Model: "test/llm", InputTokens: 1_000_000, OutputTokens: 1_000_000}
	if !p.Enrich(u) || u.CostUSD == nil || !near(*u.CostUSD, 2) {
		t.Errorf("enrich = %+v", u)
	}
	if p.Enrich(&echo.Usage{Model: "unknown/x"}) {
		t.Errorf("enrich of unknown model should report false")
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	p := Default().Clone()
	p.Set("test/tiered", Price{Rates: Rates{Input: 1, Output: 10, CacheRead: 0.1}, Tiers: []Tier{{Above: 1000, Rates: Rates{Input: 2, Output: 20}}}})

	var buf bytes.Buffer
	if err := p.Save(&buf); err != nil {
		t.Fatal(err)
	}
	q, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"test/tiered", "openai/gpt-5.2", "voyage/voyage-4", "google/gemini-2.5-pro"} {
		a, _ := p.Get(m)
		b, ok := q.Get(m)
		if !ok || a.Rates != b.Rates || len(a.Tiers) != len(b.Tiers) {
			t.Errorf("%s: %+v != %+v", m, a, b)
		}
	}
}

func TestFetch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"openai":{"name":"OpenAI","models":{"gpt-x":{"id":"gpt-x","cost":{"input":3,"output":9,"input_audio":40}},"tts":{"id":"tts"}}}}`))
	}))
	defer srv.Close()

	p, err := fetch(context.Background(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if price, ok := p.Get("openai/gpt-x"); !ok || price.Input != 3 || price.Output != 9 {
		t.Errorf("gpt-x = %+v %v", price, ok)
	}
	if _, ok := p.Get("openai/tts"); ok {
		t.Errorf("model without cost should be skipped")
	}
	if _, ok := p.Get("voyage/voyage-4"); !ok {
		t.Errorf("built-in voyage prices should be added")
	}

	if _, err := fetch(context.Background(), srv.URL+"/missing\x00"); err == nil {
		t.Errorf("expected error for bad url")
	}
}

func TestWithPricer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"model":"jev-1.13.0","answers":{"n":{"type":"noul","noul":0.9}},"usage":{"input_tokens":500000,"output_tokens":10}}`))
	}))
	defer srv.Close()

	client := echo.NewTypeSafeClient("k", "jev-latest", echo.WithBaseURL(srv.URL), echo.WithPricer(Default()))
	resp, err := client.Decide(context.Background(), "state", echo.Questions{"n": echo.Noul("?")})
	if err != nil {
		t.Fatal(err)
	}
	u := resp.Usage
	if u == nil || u.Model != "typesafe/jev-latest" || u.CostUSD == nil || !near(*u.CostUSD, 0.02) {
		t.Errorf("usage = %+v", u)
	}

	// unknown model: usage kept, no cost
	llm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
	}))
	defer llm.Close()
	cresp, err := echo.NewOpenAIClient("k", "my-finetune", echo.WithBaseURL(llm.URL), echo.WithPricer(Default())).
		Complete(context.Background(), echo.QuickMessage("x"))
	if err != nil {
		t.Fatal(err)
	}
	if cresp.Usage == nil || cresp.Usage.Model != "openai/my-finetune" || cresp.Usage.TotalTokens != 5 || cresp.Usage.CostUSD != nil {
		t.Errorf("unknown model usage = %+v", cresp.Usage)
	}
}
