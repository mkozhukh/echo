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

	options := make([]echo.CallOption, 1)
	options[0] = echo.WithMaxTokens(20000)
	if prompt != "" {
		options = append(options, echo.WithSystemMessage(prompt))
	}

	client, err = echo.NewClient(model, key, options...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating client: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	stream, err := client.StreamCall(ctx, echo.QuickMessage(message))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error calling LLM: %v\n", err)
		os.Exit(1)
	}

	for chunk := range stream.Stream {
		if chunk.Error != nil {
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", chunk.Error)
			os.Exit(1)
		}
		if chunk.Data != "" {
			fmt.Print(chunk.Data)
		}
	}
}
