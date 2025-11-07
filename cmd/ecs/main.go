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
	var model, prompt string
	flag.StringVar(&prompt, "prompt", "", "Prompt to send to the model")
	flag.StringVar(&model, "model", "", "Model in format provider/model-name")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: ec [--model provider/model] [--key api-key] message...")
		os.Exit(1)
	}

	if model == "" {
		model = os.Getenv("ECHO_MODEL")
	}

	message := strings.Join(flag.Args(), " ")

	var client echo.Client
	var err error

	options := make([]echo.CallOption, 1)
	options[0] = echo.WithMaxTokens(5000)
	if prompt != "" {
		options = append(options, echo.WithSystemMessage(prompt))
	}
	if model != "" {
		options = append(options, echo.WithModel(model))
	}

	client, err = echo.NewCommonClient(nil, options...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating client: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	stream, err := client.StreamComplete(ctx, echo.QuickMessage(message))
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
