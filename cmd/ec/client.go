package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/mkozhukh/echo"
)

// buildClient constructs an echo.Client honoring global and per-call options.
// The caller passes any extra CallOptions that are specific to the subcommand.
func buildClient(g globalOptions, extra ...echo.CallOption) (echo.Client, error) {
	opts := make([]echo.CallOption, 0, len(extra)+2)
	if g.model != "" {
		opts = append(opts, echo.WithModel(g.model))
	}
	if g.baseURL != "" {
		opts = append(opts, echo.WithBaseURL(g.baseURL))
	}
	opts = append(opts, extra...)

	client, err := echo.NewCommonClient(nil, opts...)
	if err != nil {
		return nil, fmt.Errorf("create client: %w", err)
	}
	if g.localCLI {
		echo.EnableLocalCLI(client)
	}
	if g.localLlama {
		echo.EnableLocalLlama(client)
	}
	return client, nil
}

// readPromptFromArgsOrStdin joins positional args into a prompt. If no args
// were provided and stdin is not a terminal, stdin is consumed instead. This
// lets the CLI be used both as "ec complete 'hello'" and "echo hi | ec
// complete".
func readPromptFromArgsOrStdin(args []string) (string, error) {
	if len(args) > 0 {
		return strings.Join(args, " "), nil
	}
	// Fall back to stdin when it's piped in.
	info, err := os.Stdin.Stat()
	if err == nil && (info.Mode()&os.ModeCharDevice) == 0 {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return "", fmt.Errorf("read stdin: %w", err)
		}
		text := strings.TrimRight(string(data), "\n")
		if text == "" {
			return "", fmt.Errorf("prompt is empty")
		}
		return text, nil
	}
	return "", fmt.Errorf("no prompt supplied (pass as argument or via stdin)")
}
