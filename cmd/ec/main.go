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

	var client echo.Client
	var err error

	options := make([]echo.CallOption, 0)
	if prompt != "" {
		options = append(options, echo.WithSystemMessage(prompt))
	}

	client, err = echo.NewClient(model, key, options...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating client: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	resp, err := client.Call(ctx, echo.QuickMessage(message))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error calling LLM: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(resp.Text)
}
