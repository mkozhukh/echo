package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

type embedOptions struct {
	global globalOptions
	asJSON bool
}

func newEmbedCmd() *cobra.Command {
	opts := &embedOptions{}

	cmd := &cobra.Command{
		Use:     "embed [flags] [text...]",
		Aliases: []string{"embeddings"},
		Short:   "Compute embeddings for the supplied text",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runEmbed(cmd.Context(), opts, args)
		},
	}

	addClientFlags(cmd, &opts.global)
	cmd.Flags().BoolVar(&opts.asJSON, "json", false,
		"Emit the full response as JSON instead of space-separated floats")
	return cmd
}

func runEmbed(ctx context.Context, opts *embedOptions, args []string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	text, err := readPromptFromArgsOrStdin(args)
	if err != nil {
		return err
	}

	client, err := buildClient(opts.global)
	if err != nil {
		return err
	}

	resp, err := client.GetEmbeddings(ctx, []string{text})
	if err != nil {
		return err
	}

	if opts.asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(resp)
	}

	if len(resp.Embeddings) == 0 {
		fmt.Println()
		return nil
	}
	for i, v := range resp.Embeddings[0] {
		if i > 0 {
			fmt.Print(" ")
		}
		fmt.Printf("%f", v)
	}
	fmt.Println()
	return nil
}
