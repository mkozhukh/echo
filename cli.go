package echo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
)

// CLIProvider wraps a locally installed CLI tool (e.g. Claude Code, Codex,
// Gemini) and exposes it through the echo Provider interface. The prompt is
// assembled from the message chain and forwarded to the binary in
// non-interactive mode; stdout is returned as the completion text.
//
// CLI providers execute local binaries. They are NOT registered by
// NewCommonClient automatically - callers must opt in explicitly via
// EnableLocalCLI or by constructing and SetProvider-ing instances themselves.
//
// Only the Model option has an effect on a call - temperature, max tokens,
// structured output, etc. are silently ignored because the CLI frontends do
// not expose them in a uniform way. The SystemMsg option, if provided, is
// prepended to the prompt.
type CLIProvider struct {
	// Binary is the default executable name or absolute path.
	Binary string
	// EnvVar, if non-empty, overrides Binary when the named environment
	// variable is set. This lets deployments swap in a pinned binary path
	// without recompiling.
	EnvVar string
	// ModelFlag is the flag used to pass the model name (e.g. "--model").
	// If empty, the model is omitted from the command line.
	ModelFlag string
	// ExtraArgs are appended to the command line before the prompt (when
	// PromptAsArg is true) or as the final arguments (when stdin is used).
	ExtraArgs []string
	// PromptAsArg controls how the prompt is delivered. When true the prompt
	// is appended as the final positional argument; otherwise it is piped to
	// the child process via stdin.
	PromptAsArg bool
	// WorkDir sets the working directory for the child process. When empty,
	// a fresh temporary directory is created per call and removed after the
	// process exits, which limits the files an agentic CLI can see.
	WorkDir string
}

// resolveBinary returns the effective binary path, honoring the EnvVar
// override when set.
func (p *CLIProvider) resolveBinary() string {
	if p.EnvVar != "" {
		if v := os.Getenv(p.EnvVar); v != "" {
			return v
		}
	}
	return p.Binary
}

// buildPrompt flattens a message chain into a single prompt string.
func (p *CLIProvider) buildPrompt(messages []Message, cfg CallConfig) string {
	// Resolve system message - cfg.SystemMsg overrides any in the chain.
	sys := cfg.SystemMsg
	if len(messages) > 0 && messages[0].Role == System {
		if sys == "" {
			sys = messages[0].Content
		}
		messages = messages[1:]
	}

	var sb strings.Builder
	if sys != "" {
		sb.WriteString("[system]: ")
		sb.WriteString(sys)
		sb.WriteString("\n\n")
	}
	for i, m := range messages {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString("[")
		sb.WriteString(m.Role)
		sb.WriteString("]: ")
		sb.WriteString(m.Content)
	}
	return sb.String()
}

// buildCmd constructs the exec.Cmd for the given prompt and config. The
// returned cleanup function must be invoked after the command completes; it
// removes the scratch working directory when one was created for this call.
func (p *CLIProvider) buildCmd(ctx context.Context, prompt string, cfg CallConfig) (*exec.Cmd, string, func(), error) {
	bin := p.resolveBinary()
	if bin == "" {
		return nil, "", func() {}, fmt.Errorf("cli binary is not configured")
	}

	args := append([]string{}, p.ExtraArgs...)
	if p.ModelFlag != "" && cfg.Model != "" && cfg.Model != "default" {
		args = append(args, p.ModelFlag, cfg.Model)
	}
	if p.PromptAsArg {
		args = append(args, prompt)
	}

	cmd := exec.CommandContext(ctx, bin, args...)
	if !p.PromptAsArg {
		cmd.Stdin = strings.NewReader(prompt)
	}

	cleanup := func() {}
	if p.WorkDir != "" {
		cmd.Dir = p.WorkDir
	} else {
		dir, err := os.MkdirTemp("", "echo-cli-")
		if err != nil {
			return nil, "", cleanup, fmt.Errorf("create scratch workdir: %w", err)
		}
		cmd.Dir = dir
		cleanup = func() { os.RemoveAll(dir) }
	}
	return cmd, bin, cleanup, nil
}

// call implements Provider.
func (p *CLIProvider) call(ctx context.Context, messages []Message, cfg CallConfig) (*Response, error) {
	if err := validateMessages(messages); err != nil {
		return nil, fmt.Errorf("invalid message chain: %w", err)
	}

	prompt := p.buildPrompt(messages, cfg)
	cmd, bin, cleanup, err := p.buildCmd(ctx, prompt, cfg)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		errText := strings.TrimSpace(stderr.String())
		if errText != "" {
			return nil, fmt.Errorf("cli %q failed: %w: %s", bin, err, errText)
		}
		return nil, fmt.Errorf("cli %q failed: %w", bin, err)
	}

	text := strings.TrimRight(stdout.String(), "\n")
	return &Response{
		Text: text,
		Metadata: Metadata{
			"provider": "cli",
			"binary":   bin,
			"model":    cfg.Model,
		},
	}, nil
}

// streamCall implements Provider. The CLI is executed fully and its output is
// delivered as a single chunk. CLI frontends do not expose uniform streaming
// APIs, so we intentionally keep the wrapper simple.
func (p *CLIProvider) streamCall(ctx context.Context, messages []Message, cfg CallConfig) (*StreamResponse, error) {
	resp, err := p.call(ctx, messages, cfg)
	if err != nil {
		return nil, err
	}

	ch := make(chan StreamChunk, 3)
	meta := resp.Metadata
	ch <- StreamChunk{Meta: &meta}
	ch <- StreamChunk{Data: resp.Text}
	ch <- StreamChunk{Error: nil}
	close(ch)
	return &StreamResponse{Stream: ch}, nil
}

// getEmbeddings is not supported by CLI providers.
func (p *CLIProvider) getEmbeddings(ctx context.Context, texts []string, cfg CallConfig) (*EmbeddingResponse, error) {
	return nil, fmt.Errorf("embeddings are not supported by cli providers")
}

// reRank is not supported by CLI providers.
func (p *CLIProvider) reRank(ctx context.Context, query string, documents []string, cfg CallConfig) (*RerankResponse, error) {
	return nil, fmt.Errorf("rerank is not supported by cli providers")
}

// parseCompletionRequest accepts OpenAI-format completion payloads so that a
// CLI provider can sit behind the proxy helpers just like the HTTP providers.
func (p *CLIProvider) parseCompletionRequest(req *http.Request) (*CompletionRequest, error) {
	var r CompletionRequest
	if err := json.NewDecoder(req.Body).Decode(&r); err != nil {
		return nil, fmt.Errorf("failed to parse cli completion request: %w", err)
	}
	return &r, nil
}

func (p *CLIProvider) parseEmbeddingRequest(req *http.Request) (*EmbeddingRequest, error) {
	return nil, fmt.Errorf("embeddings are not supported by cli providers")
}

func (p *CLIProvider) parseRerankRequest(req *http.Request) (*RerankRequest, error) {
	return nil, fmt.Errorf("rerank is not supported by cli providers")
}

func (p *CLIProvider) buildCompletionRequest(ctx context.Context, req *CompletionRequest, cfg CallConfig) (*CompletionResponse, error) {
	msgs := make([]Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		role := m.Role
		if role == "assistant" {
			role = Agent
		}
		msgs = append(msgs, Message{Role: role, Content: m.Content})
	}

	resp, err := p.call(ctx, msgs, cfg)
	if err != nil {
		return nil, err
	}

	out := &CompletionResponse{
		Object: "chat.completion",
		Model:  cfg.Model,
		Choices: make([]struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason,omitempty"`
		}, 1),
	}
	out.Choices[0].Message.Role = "assistant"
	out.Choices[0].Message.Content = resp.Text
	out.Choices[0].FinishReason = "stop"
	return out, nil
}

func (p *CLIProvider) buildEmbeddingRequest(ctx context.Context, req *EmbeddingRequest, cfg CallConfig) (*UnifiedEmbeddingResponse, error) {
	return nil, fmt.Errorf("embeddings are not supported by cli providers")
}

func (p *CLIProvider) buildRerankRequest(ctx context.Context, req *RerankRequest, cfg CallConfig) (*UnifiedRerankResponse, error) {
	return nil, fmt.Errorf("rerank is not supported by cli providers")
}

func (p *CLIProvider) writeCompletionResponse(w http.ResponseWriter, resp *CompletionResponse) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(resp)
}

func (p *CLIProvider) writeEmbeddingResponse(w http.ResponseWriter, resp *UnifiedEmbeddingResponse) error {
	return fmt.Errorf("embeddings are not supported by cli providers")
}

func (p *CLIProvider) writeRerankResponse(w http.ResponseWriter, resp *UnifiedRerankResponse) error {
	return fmt.Errorf("rerank is not supported by cli providers")
}

// NewClaudeCLIProvider returns a provider wired for the Claude Code CLI.
// Override the binary path via the ECHO_CLAUDE_CLI_PATH environment variable.
func NewClaudeCLIProvider() *CLIProvider {
	return &CLIProvider{
		Binary:    "claude",
		EnvVar:    "ECHO_CLAUDE_CLI_PATH",
		ModelFlag: "--model",
		ExtraArgs: []string{"-p", "--output-format", "text"},
	}
}

// NewCodexCLIProvider returns a provider wired for the OpenAI Codex CLI.
// Override the binary path via the ECHO_CODEX_CLI_PATH environment variable.
func NewCodexCLIProvider() *CLIProvider {
	return &CLIProvider{
		Binary:      "codex",
		EnvVar:      "ECHO_CODEX_CLI_PATH",
		ModelFlag:   "-m",
		ExtraArgs:   []string{"exec"},
		PromptAsArg: true,
	}
}

// NewGeminiCLIProvider returns a provider wired for the Gemini CLI.
// Override the binary path via the ECHO_GEMINI_CLI_PATH environment variable.
func NewGeminiCLIProvider() *CLIProvider {
	return &CLIProvider{
		Binary:    "gemini",
		EnvVar:    "ECHO_GEMINI_CLI_PATH",
		ModelFlag: "-m",
		ExtraArgs: []string{"-p"},
		// gemini's -p expects the prompt as its argument.
		PromptAsArg: true,
	}
}

// NewOpenCodeCLIProvider returns a provider wired for the opencode CLI.
// Override the binary path via the ECHO_OPENCODE_CLI_PATH environment variable.
func NewOpenCodeCLIProvider() *CLIProvider {
	return &CLIProvider{
		Binary:      "opencode",
		EnvVar:      "ECHO_OPENCODE_CLI_PATH",
		ModelFlag:   "--model",
		ExtraArgs:   []string{"run"},
		PromptAsArg: true,
	}
}

// EnableLocalLlama registers an OpenAI-compatible provider targeting a local
// llama.cpp server under the "llama" name. The default host is
// http://127.0.0.1:8080; set ECHO_LLAMA_URL to override (origin only, e.g.
// http://host:9090). Both chat completions and embeddings are served from
// the same host using the standard /v1/chat/completions and /v1/embeddings
// paths.
//
// Models are referenced as "llama/<any-tag>" - llama.cpp serves whichever
// model the server was started with, so the tag is cosmetic unless the
// server was launched with multiple models.
func EnableLocalLlama(client Client) {
	host := os.Getenv("ECHO_LLAMA_URL")
	if host == "" {
		host = "http://127.0.0.1:8080"
	}
	client.SetProvider("llama", &OpenAIProvider{Host: host})
}

// EnableLocalCLI registers the local CLI providers on the given client.
// Because these providers execute local binaries, they are intentionally NOT
// registered by NewCommonClient - callers must opt in explicitly.
//
// After this call the following provider names are available and can be
// referenced through the standard "provider/model" syntax:
//
//	claude-cli/<model>     e.g. claude-cli/opus
//	codex-cli/<model>      e.g. codex-cli/gpt-5
//	gemini-cli/<model>     e.g. gemini-cli/pro
//	opencode-cli/<model>   e.g. opencode-cli/anthropic/claude-sonnet-4
//
// Binary paths can be overridden via the environment variables
// ECHO_CLAUDE_CLI_PATH, ECHO_CODEX_CLI_PATH, ECHO_GEMINI_CLI_PATH, and
// ECHO_OPENCODE_CLI_PATH.
func EnableLocalCLI(client Client) {
	client.SetProvider("claude-cli", NewClaudeCLIProvider())
	client.SetProvider("codex-cli", NewCodexCLIProvider())
	client.SetProvider("gemini-cli", NewGeminiCLIProvider())
	client.SetProvider("opencode-cli", NewOpenCodeCLIProvider())
}
