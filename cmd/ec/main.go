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
	var model, key, prompt string
	flag.StringVar(&prompt, "prompt", "", "Prompt to send to the model")
	flag.StringVar(&model, "model", "", "Model in format provider/model-name")
	flag.StringVar(&key, "key", "", "API key for the provider")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: ec [--model provider/model] [--key api-key] message...")
		os.Exit(1)
	}

	message := strings.Join(flag.Args(), " ")

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

	options := make([]echo.CallOption, 0)
	if prompt != "" {
		options = append(options, echo.WithSystemMessage(prompt))
	}

	switch provider {
	case "openai":
		client = echo.NewOpenAIClient(key, modelName, options...)
	case "anthropic":
		client = echo.NewAnthropicClient(key, modelName, options...)
	case "gemini":
		client = echo.NewGeminiClient(key, modelName, options...)
	default:
		fmt.Fprintf(os.Stderr, "Unknown provider: %s\n", provider)
		os.Exit(1)
	}

	ctx := context.Background()
	resp, err := client.Call(ctx, message)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error calling LLM: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(resp.Text)
}
