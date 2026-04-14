package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/mkozhukh/echo"
	"github.com/mkozhukh/mdterm"
	"github.com/spf13/cobra"
)

type completeOptions struct {
	global globalOptions

	system         string
	temperature    float32
	temperatureSet bool
	maxTokens      int
	reasoning      string
	endpoint       string
	structuredName string
	structuredFile string
	pretty         bool
	noStream       bool
	storeData      bool
	storeDataSet   bool
}

func newCompleteCmd() *cobra.Command {
	opts := &completeOptions{}

	cmd := &cobra.Command{
		Use:     "complete [flags] [prompt...]",
		Aliases: []string{"c", "chat"},
		Short:   "Send a prompt to the completion endpoint",
		Long: "Send a prompt to the configured LLM completion endpoint and " +
			"stream the response to stdout.\n\n" +
			"The prompt may be supplied as positional arguments or piped via stdin.",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Record which flags the user actually set so we only forward
			// non-default values into the CallConfig.
			opts.temperatureSet = cmd.Flags().Changed("temperature")
			opts.storeDataSet = cmd.Flags().Changed("store-data")
			return runComplete(cmd.Context(), opts, args)
		},
	}

	addClientFlags(cmd, &opts.global)
	cmd.Flags().StringVarP(&opts.system, "system", "s", "",
		"System message / instructions")
	cmd.Flags().Float32VarP(&opts.temperature, "temperature", "t", 0,
		"Sampling temperature")
	cmd.Flags().IntVar(&opts.maxTokens, "max-tokens", 5000,
		"Maximum number of tokens to generate")
	cmd.Flags().StringVar(&opts.reasoning, "reasoning", "",
		"Reasoning effort: low, medium, high")
	cmd.Flags().StringVar(&opts.endpoint, "endpoint", "",
		"Override the request endpoint path")
	cmd.Flags().StringVar(&opts.structuredName, "schema-name", "",
		"Name for the structured-output schema (requires --schema)")
	cmd.Flags().StringVar(&opts.structuredFile, "schema", "",
		"Path to a JSON Schema file to request structured output")
	cmd.Flags().BoolVar(&opts.pretty, "pretty", false,
		"Render streamed markdown to the terminal via mdterm")
	cmd.Flags().BoolVar(&opts.noStream, "no-stream", false,
		"Buffer the full response before printing instead of streaming")
	cmd.Flags().BoolVar(&opts.storeData, "store-data", false,
		"xAI only: let the provider store conversation data server-side")

	return cmd
}

func runComplete(ctx context.Context, opts *completeOptions, args []string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	// Cancel the request on Ctrl-C so the HTTP stream unwinds cleanly.
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	prompt, err := readPromptFromArgsOrStdin(args)
	if err != nil {
		return err
	}

	callOpts, err := buildCallOptions(opts)
	if err != nil {
		return err
	}

	client, err := buildClient(opts.global)
	if err != nil {
		return err
	}

	msgs := echo.QuickMessage(prompt)

	// Choose the output sink. --pretty takes precedence; mdterm's parser
	// implements io.Writer and flushes block-by-block.
	out := selectOutput(opts.pretty)
	if closer, ok := out.(io.Closer); ok {
		defer closer.Close()
	}

	if opts.noStream {
		resp, err := client.Complete(ctx, msgs, callOpts...)
		if err != nil {
			return err
		}
		if _, err := io.WriteString(out, resp.Text); err != nil {
			return err
		}
		return finalizeOutput(out, opts.pretty)
	}

	stream, err := client.StreamComplete(ctx, msgs, callOpts...)
	if err != nil {
		return err
	}

	// Minimal buffering: forward each chunk to the writer as it arrives.
	for chunk := range stream.Stream {
		if chunk.Error != nil {
			return fmt.Errorf("stream: %w", chunk.Error)
		}
		if chunk.Data == "" {
			continue
		}
		if _, err := io.WriteString(out, chunk.Data); err != nil {
			return err
		}
	}
	return finalizeOutput(out, opts.pretty)
}

// finalizeOutput emits the trailing whitespace needed to flush the underlying
// writer. mdterm buffers until a block boundary (blank line), so in pretty
// mode we write two newlines; otherwise a single newline is enough to give
// the user a clean prompt.
func finalizeOutput(out io.Writer, pretty bool) error {
	tail := "\n"
	if pretty {
		tail = "\n\n"
	}
	_, err := io.WriteString(out, tail)
	return err
}

// selectOutput returns the writer that consumes streamed completion text.
// When pretty rendering is requested we route through mdterm, otherwise we
// write directly to stdout so bytes hit the terminal as soon as they arrive.
func selectOutput(pretty bool) io.Writer {
	if pretty {
		return mdterm.New(os.Stdout)
	}
	return os.Stdout
}

// buildCallOptions translates completeOptions into echo.CallOption values,
// omitting defaults so that the library's own defaults remain in effect.
func buildCallOptions(opts *completeOptions) ([]echo.CallOption, error) {
	var callOpts []echo.CallOption
	if opts.maxTokens > 0 {
		callOpts = append(callOpts, echo.WithMaxTokens(opts.maxTokens))
	}
	if opts.system != "" {
		callOpts = append(callOpts, echo.WithSystemMessage(opts.system))
	}
	if opts.temperatureSet {
		callOpts = append(callOpts, echo.WithTemperature(opts.temperature))
	}
	if opts.reasoning != "" {
		callOpts = append(callOpts, echo.WithReasoningEffort(opts.reasoning))
	}
	if opts.endpoint != "" {
		callOpts = append(callOpts, echo.WithEndPoint(opts.endpoint))
	}
	if opts.storeDataSet {
		callOpts = append(callOpts, echo.WithStoreData(opts.storeData))
	}
	if opts.structuredFile != "" {
		schema, err := loadSchema(opts.structuredFile)
		if err != nil {
			return nil, err
		}
		name := opts.structuredName
		if name == "" {
			name = "response"
		}
		callOpts = append(callOpts, echo.WithStructuredOutput(name, schema))
	}
	return callOpts, nil
}

func loadSchema(path string) (any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read schema: %w", err)
	}
	var schema any
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("parse schema %q: %w", path, err)
	}
	return schema, nil
}
