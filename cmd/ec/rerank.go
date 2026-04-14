package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
)

type rerankOptions struct {
	global   globalOptions
	query    string
	docsFile string
	asJSON   bool
}

func newRerankCmd() *cobra.Command {
	opts := &rerankOptions{}

	cmd := &cobra.Command{
		Use:   "rerank [flags] [documents...]",
		Short: "Rerank documents by relevance to a query",
		Long: "Rerank a list of documents against the --query. Documents can " +
			"be passed as positional arguments, read from a file via --docs, " +
			"or piped one-per-line via stdin.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRerank(cmd.Context(), opts, args)
		},
	}

	addClientFlags(cmd, &opts.global)
	cmd.Flags().StringVarP(&opts.query, "query", "q", "",
		"Query to rank documents against (required)")
	cmd.Flags().StringVar(&opts.docsFile, "docs", "",
		"Path to a file with one document per line")
	cmd.Flags().BoolVar(&opts.asJSON, "json", false,
		"Emit the full response as JSON")
	_ = cmd.MarkFlagRequired("query")
	return cmd
}

func runRerank(ctx context.Context, opts *rerankOptions, args []string) error {
	if ctx == nil {
		ctx = context.Background()
	}

	docs, err := collectDocuments(opts.docsFile, args)
	if err != nil {
		return err
	}
	if len(docs) == 0 {
		return fmt.Errorf("no documents supplied")
	}

	client, err := buildClient(opts.global)
	if err != nil {
		return err
	}

	resp, err := client.ReRank(ctx, opts.query, docs)
	if err != nil {
		return err
	}

	if opts.asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(resp)
	}

	for i, score := range resp.Scores {
		fmt.Printf("%d\t%f\n", i, score)
	}
	return nil
}

// collectDocuments reads documents from the requested source. Positional args
// win, then --docs, then stdin if it's piped in.
func collectDocuments(docsFile string, args []string) ([]string, error) {
	if len(args) > 0 {
		return args, nil
	}
	if docsFile != "" {
		f, err := os.Open(docsFile)
		if err != nil {
			return nil, fmt.Errorf("open docs: %w", err)
		}
		defer f.Close()
		return scanLines(f)
	}
	info, err := os.Stdin.Stat()
	if err == nil && (info.Mode()&os.ModeCharDevice) == 0 {
		return scanLines(os.Stdin)
	}
	return nil, nil
}

func scanLines(r io.Reader) ([]string, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return lines, nil
}
