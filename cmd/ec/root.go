package main

import (
	"github.com/spf13/cobra"
)

// globalOptions holds flags shared by subcommands that build an echo client.
type globalOptions struct {
	model   string
	baseURL string
}

// addClientFlags registers flags common to every subcommand that instantiates
// an echo client. The model flag also falls back to the ECHO_MODEL env var;
// see resolveModel in client.go.
func addClientFlags(cmd *cobra.Command, o *globalOptions) {
	cmd.Flags().StringVarP(&o.model, "model", "m", "",
		"Model in format provider/model (env: ECHO_MODEL)")
	cmd.Flags().StringVar(&o.baseURL, "base-url", "",
		"Override the provider base URL")
}

func newRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:           "ec",
		Short:         "Echo LLM command-line client",
		Long:          "ec is a thin command-line wrapper around the echo LLM adapter library.",
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	root.AddCommand(newCompleteCmd())
	root.AddCommand(newEmbedCmd())
	root.AddCommand(newRerankCmd())
	root.AddCommand(newVersionCmd())
	return root
}
