package echo

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChoiceCriteriaKeepsOrder(t *testing.T) {
	q := Choice("Main request?", Option{"zeta", "last"}, Option{"alpha", "first"})
	b, err := json.Marshal(q)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"type":"choice","instructions":"Main request?","criteria":{"zeta":"last","alpha":"first"}}`
	if string(b) != want {
		t.Errorf("got %s, want %s", b, want)
	}
}

func TestQuestionJSON(t *testing.T) {
	b, _ := json.Marshal(Questions{
		"s": Score("How bad?", "low", "high"),
		"n": Noul("Is it bad").WithCriteria("yes means bad"),
	})
	want := `{"n":{"type":"noul","instructions":"Is it bad","criteria":"yes means bad"},"s":{"type":"score","instructions":"How bad?","criteria":["low","high"]}}`
	if string(b) != want {
		t.Errorf("got %s, want %s", b, want)
	}
}

func TestValidateQuestions(t *testing.T) {
	bad := []Questions{
		nil,
		{"x": {Type: "unknown", Instructions: "?"}},
		{"x": {Type: QuestionNoul}},
		{"x": {Type: QuestionChoice, Instructions: "?"}},
		{"": Noul("?")},
	}
	for i, q := range bad {
		if err := validateQuestions(q); err == nil {
			t.Errorf("case %d: expected error", i)
		}
	}
}

func TestMockDecide(t *testing.T) {
	client, _ := NewCommonClient(nil, WithModel("mock/test"))
	resp, err := client.Decide(context.Background(), "ticket text", Questions{
		"type":     Choice("Main request?", Option{"refund", "money back"}, Option{"bug", "defect"}),
		"anger":    Score("How frustrated?", "calm", "annoyed", "furious"),
		"escalate": Noul("Customer threatens to cancel"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if c, conf := resp.Choice("type"); c != "refund" || conf != 1 {
		t.Errorf("choice = %q %v", c, conf)
	}
	if s := resp.Score("anger"); s != 0 {
		t.Errorf("score = %v", s)
	}
	if l := resp.Answers["anger"].Legend["2"]; l != "furious" {
		t.Errorf("legend = %q", l)
	}
	if n := resp.Noul("escalate"); n != 0.5 {
		t.Errorf("noul = %v", n)
	}
}

func TestDecideUnsupportedProvider(t *testing.T) {
	client, _ := NewCommonClient(map[string]string{"openai": "key"}, WithModel("openai/gpt-5"))
	_, err := client.Decide(context.Background(), "x", Questions{"n": Noul("?")})
	if err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Errorf("expected not supported error, got %v", err)
	}
}

func TestTypeSafeDecide(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("auth header = %q", r.Header.Get("Authorization"))
		}
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &got)
		w.Write([]byte(`{
			"model": "jev-1.13.0",
			"answers": {
				"type": {"type": "choice", "choice": "bug", "confidence": 0.9, "probabilities": {"refund": 0.1, "bug": 0.9}},
				"anger": {"type": "score", "score": 0.67, "confidence": 0.5, "legend": {"0": "calm", "1": "annoyed", "2": "furious"}, "probabilities": {"0": 0.33, "1": 0.67, "2": 0}},
				"escalate": {"type": "noul", "noul": 0.99}
			},
			"usage": {"input_tokens": 428, "output_tokens": 73}
		}`))
	}))
	defer srv.Close()

	client := NewTypeSafeClient("secret", "jev-latest", WithBaseURL(srv.URL))
	resp, err := client.Decide(context.Background(), map[string]string{"ticket": "app crashes"}, Questions{
		"type":     Choice("Main request?", Option{"refund", "money back"}, Option{"bug", "defect"}),
		"anger":    Score("How frustrated?", "calm", "annoyed", "furious"),
		"escalate": Noul("Customer threatens to cancel"),
	})
	if err != nil {
		t.Fatal(err)
	}

	if got["model"] != "jev-latest" {
		t.Errorf("request model = %v", got["model"])
	}
	if st, _ := got["state"].(map[string]any); st["ticket"] != "app crashes" {
		t.Errorf("request state = %v", got["state"])
	}
	if qs, _ := got["questions"].(map[string]any); len(qs) != 3 {
		t.Errorf("request questions = %v", got["questions"])
	}

	if c, conf := resp.Choice("type"); c != "bug" || conf != 0.9 {
		t.Errorf("choice = %q %v", c, conf)
	}
	if s := resp.Score("anger"); s != 0.67 {
		t.Errorf("score = %v", s)
	}
	if n := resp.Noul("escalate"); n != 0.99 {
		t.Errorf("noul = %v", n)
	}
	if resp.Answers["escalate"].Confidence != nil {
		t.Errorf("noul confidence should be nil")
	}
	if resp.Model != "jev-1.13.0" || resp.Metadata["input_tokens"] != 428 || resp.Metadata["output_tokens"] != 73 {
		t.Errorf("model/metadata = %v %v", resp.Model, resp.Metadata)
	}
}

func TestTypeSafeAlias(t *testing.T) {
	client := NewTypeSafeClient("k", "")
	cc := client.(*CommonClient)
	p, m, _, err := cc.resolveProviderAndModel("typesafe/best")
	if err != nil || p != "typesafe" || m != "jev-latest" {
		t.Errorf("got %q %q %v", p, m, err)
	}
	_, err = client.Complete(context.Background(), QuickMessage("hi"), WithModel("typesafe/jev-latest"))
	if err == nil {
		t.Errorf("expected Complete to be unsupported")
	}
}
