package echo

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFakeCLI writes a tiny shell script to dir that echoes the args it
// received followed by stdin, so tests can assert how the CLIProvider wired
// up its command line.
func writeFakeCLI(t *testing.T, dir, name string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	script := `#!/bin/sh
echo "args: $@"
echo "--- stdin ---"
cat
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake cli: %v", err)
	}
	return path
}

func TestCLIProvider_StdinPrompt(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeCLI(t, dir, "fake")

	p := &CLIProvider{
		Binary:    bin,
		ModelFlag: "--model",
		ExtraArgs: []string{"-p"},
	}

	ctx := context.Background()
	resp, err := p.call(ctx, QuickMessage("hello world"), CallConfig{Model: "opus"})
	if err != nil {
		t.Fatalf("call: %v", err)
	}

	if !strings.Contains(resp.Text, "args: -p --model opus") {
		t.Errorf("expected args in output, got: %q", resp.Text)
	}
	if !strings.Contains(resp.Text, "[user]: hello world") {
		t.Errorf("expected prompt delivered via stdin, got: %q", resp.Text)
	}
	if resp.Metadata["binary"] != bin {
		t.Errorf("expected binary metadata %q, got %v", bin, resp.Metadata["binary"])
	}
}

func TestCLIProvider_PromptAsArg(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeCLI(t, dir, "fake")

	p := &CLIProvider{
		Binary:      bin,
		ModelFlag:   "-m",
		ExtraArgs:   []string{"exec"},
		PromptAsArg: true,
	}

	ctx := context.Background()
	resp, err := p.call(ctx, QuickMessage("hi"), CallConfig{Model: "gpt-5"})
	if err != nil {
		t.Fatalf("call: %v", err)
	}

	// Prompt should arrive as a positional arg, not via stdin.
	if !strings.Contains(resp.Text, "args: exec -m gpt-5 [user]: hi") {
		t.Errorf("expected prompt as arg, got: %q", resp.Text)
	}
}

func TestCLIProvider_EnvVarOverride(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeCLI(t, dir, "override")

	t.Setenv("ECHO_TEST_CLI_PATH", bin)

	p := &CLIProvider{
		Binary:    "/nonexistent/claude",
		EnvVar:    "ECHO_TEST_CLI_PATH",
		ModelFlag: "--model",
	}

	ctx := context.Background()
	resp, err := p.call(ctx, QuickMessage("ping"), CallConfig{Model: "haiku"})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(resp.Text, "--model haiku") {
		t.Errorf("expected --model in output, got: %q", resp.Text)
	}
}

func TestCLIProvider_SystemMessage(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeCLI(t, dir, "fake")

	p := &CLIProvider{Binary: bin}

	ctx := context.Background()
	msgs := []Message{
		{Role: System, Content: "be terse"},
		{Role: User, Content: "hi"},
	}
	resp, err := p.call(ctx, msgs, CallConfig{Model: "opus"})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(resp.Text, "[system]: be terse") {
		t.Errorf("expected system message in prompt, got: %q", resp.Text)
	}
	if !strings.Contains(resp.Text, "[user]: hi") {
		t.Errorf("expected user message in prompt, got: %q", resp.Text)
	}
}

func TestCLIProvider_CallConfigSystemOverride(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeCLI(t, dir, "fake")

	p := &CLIProvider{Binary: bin}

	ctx := context.Background()
	resp, err := p.call(ctx, QuickMessage("hi"), CallConfig{SystemMsg: "override"})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(resp.Text, "[system]: override") {
		t.Errorf("expected system override in prompt, got: %q", resp.Text)
	}
}

func TestCLIProvider_FailurePropagatesStderr(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "broken")
	script := `#!/bin/sh
echo "boom" >&2
exit 2
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write broken cli: %v", err)
	}

	p := &CLIProvider{Binary: path}
	_, err := p.call(context.Background(), QuickMessage("hi"), CallConfig{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Errorf("expected stderr in error, got: %v", err)
	}
}

func TestEnableLocalCLI(t *testing.T) {
	client, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	EnableLocalCLI(client)

	cc := client.(*CommonClient)
	for _, name := range []string{"claude-cli", "codex-cli", "gemini-cli"} {
		if _, ok := cc.providerMap[name]; !ok {
			t.Errorf("expected %s to be registered", name)
		}
	}

	// And must NOT be registered by default in NewCommonClient.
	base, err := NewCommonClient(map[string]string{})
	if err != nil {
		t.Fatalf("NewCommonClient: %v", err)
	}
	bc := base.(*CommonClient)
	for _, name := range []string{"claude-cli", "codex-cli", "gemini-cli"} {
		if _, ok := bc.providerMap[name]; ok {
			t.Errorf("%s should not be auto-registered", name)
		}
	}
}

func TestCLIProvider_RoutingThroughClient(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeCLI(t, dir, "fake")

	client, err := NewClient(WithModel("claude-cli/opus"))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	client.SetProvider("claude-cli", &CLIProvider{
		Binary:    bin,
		ModelFlag: "--model",
	})

	resp, err := client.Complete(context.Background(), QuickMessage("hi"))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !strings.Contains(resp.Text, "--model opus") {
		t.Errorf("expected model routed into CLI flag, got: %q", resp.Text)
	}
}
