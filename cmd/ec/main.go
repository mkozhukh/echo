package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/mkozhukh/echo"
)

func main() {
	var model, key string
	flag.StringVar(&model, "model", "", "Model in format provider/model-name")
	flag.StringVar(&key, "key", "", "API key for the provider")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: ec [--model provider/model] [--key api-key] \"prompt\"")
		os.Exit(1)
	}

	prompt := flag.Arg(0)

	if model == "" {
		if envModel := os.Getenv("ECHO_MODEL"); envModel != "" {
			model = envModel
		}
	}

	if key == "" {
		if envKey := os.Getenv("ECHO_KEY"); envKey != "" {
			key = envKey
		}
	}

	parts := strings.SplitN(model, "/", 2)
	if len(parts) != 2 {
		fmt.Fprintf(os.Stderr, "Invalid model format: %s. Expected provider/model-name\n", model)
		os.Exit(1)
	}

	provider, modelName := parts[0], parts[1]

	var client echo.Client
	var err error

	switch provider {
	case "openai":
		client = echo.NewOpenAIClient(key, modelName)
	case "anthropic":
		client = echo.NewAnthropicClient(key, modelName)
	case "gemini":
		client = echo.NewGeminiClient(key, modelName)
	default:
		fmt.Fprintf(os.Stderr, "Unknown provider: %s\n", provider)
		os.Exit(1)
	}

	ctx := context.Background()
	resp, err := client.Call(ctx, prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error calling LLM: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(resp.Text)
}
