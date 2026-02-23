package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	echo "github.com/mkozhukh/echo"
)

type providerTest struct {
	name   string
	envKey string
	tests  []apiTest
}

type apiTest struct {
	name string
	fn   func(ctx context.Context, client echo.Client) error
}

func main() {
	// Optional: filter by provider name(s) via CLI args
	// Usage: go run ./cmd/e2e/ [provider...]
	// Example: go run ./cmd/e2e/ openai voyage
	filter := map[string]bool{}
	for _, arg := range os.Args[1:] {
		filter[arg] = true
	}

	client, err := echo.NewCommonClient(nil)
	if err != nil {
		fmt.Printf("FAIL: failed to create client: %v\n", err)
		os.Exit(1)
	}

	providers := []providerTest{
		{
			name:   "openai",
			envKey: "OPENAI_API_KEY",
			tests: []apiTest{
				{"Complete", testComplete("openai/gpt-4.1-nano")},
				{"StreamComplete", testStreamComplete("openai/gpt-4.1-nano")},
				{"GetEmbeddings", testGetEmbeddings("openai/text-embedding-3-small")},
			},
		},
		{
			name:   "anthropic",
			envKey: "ANTHROPIC_API_KEY",
			tests: []apiTest{
				{"Complete", testComplete("anthropic/claude-haiku-4-5")},
				{"StreamComplete", testStreamComplete("anthropic/claude-haiku-4-5")},
			},
		},
		{
			name:   "google",
			envKey: "GOOGLE_API_KEY",
			tests: []apiTest{
				{"Complete", testComplete("google/gemini-2.5-flash-lite")},
				{"StreamComplete", testStreamComplete("google/gemini-2.5-flash-lite")},
				{"GetEmbeddings", testGetEmbeddings("google/gemini-embedding-001")},
			},
		},
		{
			name:   "voyage",
			envKey: "VOYAGE_API_KEY",
			tests: []apiTest{
				{"GetEmbeddings", testGetEmbeddings("voyage/voyage-3-lite")},
				{"GetEmbeddings (batch)", testGetEmbeddingsBatch("voyage/voyage-3-lite")},
				{"ReRank", testReRank("voyage/rerank-2")},
			},
		},
		{
			name:   "xai",
			envKey: "XAI_API_KEY",
			tests: []apiTest{
				{"Complete", testComplete("xai/grok-4-1-fast-non-reasoning")},
				{"StreamComplete", testStreamComplete("xai/grok-4-1-fast-non-reasoning")},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	failed := false
	for _, p := range providers {
		if len(filter) > 0 && !filter[p.name] {
			continue
		}
		if os.Getenv(p.envKey) == "" {
			fmt.Printf("SKIP: %s (%s not set)\n", p.name, p.envKey)
			continue
		}

		for _, t := range p.tests {
			label := p.name + "/" + t.name
			if err := t.fn(ctx, client); err != nil {
				fmt.Printf("FAIL: %s - %v\n", label, err)
				failed = true
			} else {
				fmt.Printf("  OK: %s\n", label)
			}
		}
	}

	if failed {
		os.Exit(1)
	}
}

func testComplete(model string) func(ctx context.Context, client echo.Client) error {
	return func(ctx context.Context, client echo.Client) error {
		resp, err := client.Complete(ctx,
			echo.QuickMessage("Reply with one word: hello"),
			echo.WithModel(model),
			echo.WithMaxTokens(16),
		)
		if err != nil {
			return err
		}
		if strings.TrimSpace(resp.Text) == "" {
			return fmt.Errorf("empty response text")
		}
		return nil
	}
}

func testStreamComplete(model string) func(ctx context.Context, client echo.Client) error {
	return func(ctx context.Context, client echo.Client) error {
		resp, err := client.StreamComplete(ctx,
			echo.QuickMessage("Reply with one word: hello"),
			echo.WithModel(model),
			echo.WithMaxTokens(16),
		)
		if err != nil {
			return err
		}
		var text strings.Builder
		for chunk := range resp.Stream {
			if chunk.Error != nil {
				return chunk.Error
			}
			text.WriteString(chunk.Data)
		}
		if strings.TrimSpace(text.String()) == "" {
			return fmt.Errorf("empty stream response")
		}
		return nil
	}
}

func testGetEmbeddings(model string) func(ctx context.Context, client echo.Client) error {
	return func(ctx context.Context, client echo.Client) error {
		resp, err := client.GetEmbeddings(ctx,
			[]string{"hello world"},
			echo.WithModel(model),
		)
		if err != nil {
			return err
		}
		if len(resp.Embeddings) == 0 || len(resp.Embeddings[0]) == 0 {
			return fmt.Errorf("empty embeddings")
		}
		return nil
	}
}

func testGetEmbeddingsBatch(model string) func(ctx context.Context, client echo.Client) error {
	return func(ctx context.Context, client echo.Client) error {
		resp, err := client.GetEmbeddings(ctx,
			[]string{"hello", "world"},
			echo.WithModel(model),
		)
		if err != nil {
			return err
		}
		if len(resp.Embeddings) != 2 {
			return fmt.Errorf("expected 2 embeddings, got %d", len(resp.Embeddings))
		}
		if len(resp.Embeddings[0]) == 0 || len(resp.Embeddings[1]) == 0 {
			return fmt.Errorf("empty embeddings in batch")
		}
		return nil
	}
}

func testReRank(model string) func(ctx context.Context, client echo.Client) error {
	return func(ctx context.Context, client echo.Client) error {
		resp, err := client.ReRank(ctx,
			"capital of France",
			[]string{"Paris is the capital of France", "Berlin is in Germany"},
			echo.WithModel(model),
		)
		if err != nil {
			return err
		}
		if len(resp.Scores) != 2 {
			return fmt.Errorf("expected 2 scores, got %d", len(resp.Scores))
		}
		return nil
	}
}
