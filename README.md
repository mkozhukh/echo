# Echo - Simple LLM Client Library

A lightweight Go library for interacting with various LLM providers with a simple, unified API.

## Supported Providers

- OpenAI
- Anthropic
- Google Gemini

## Installation

```bash
go get github.com/mkozhukh/echo
``` 

## Quick Start

### Universal Client (Recommended)

The `NewClient` function provides a unified way to create clients for any provider:

```go
package main

import (
    "context"
    "fmt"
    "github.com/mkozhukh/echo"
)

func main() {
    ctx := context.Background()

    // Create client with provider/model format
    client, err := echo.NewClient("openai/gpt-4.1", "your-api-key")
    if err != nil {
        panic(err)
    }
    
    // Simple call using QuickMessage helper
    resp, err := client.Call(ctx, echo.QuickMessage("Hello, how are you?"))
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Text)
}
```

### Model Aliases

Use convenient aliases instead of full model names:

```go
// Quality tiers available for each provider:
// - best: Highest quality model
// - balanced: Good balance of quality and speed
// - light: Fast and economical

client, _ := echo.NewClient("openai/best", "")      // Uses gpt-4.1
client, _ := echo.NewClient("anthropic/balanced", "") // Uses claude-sonnet-4
client, _ := echo.NewClient("gemini/light", "")      // Uses gemini-2.5-flash
```

### Environment Variables

The library supports flexible environment variable configuration:

```go
// Set default model and API key
os.Setenv("ECHO_MODEL", "anthropic/balanced")
os.Setenv("ECHO_KEY", "your-api-key")

// Create client without parameters - uses env vars
client, _ := echo.NewClient("", "")

// Or use provider-specific API keys
os.Setenv("OPENAI_API_KEY", "your-openai-key")
os.Setenv("ANTHROPIC_API_KEY", "your-anthropic-key")
os.Setenv("GEMINI_API_KEY", "your-gemini-key")

// API key is automatically selected based on provider
client, _ := echo.NewClient("openai/gpt-4.1", "")
```

## Message Chains

### Simple Messages

For basic single-message prompts, use the `QuickMessage` helper:

```go
// Simple user message
resp, _ := client.Call(ctx, echo.QuickMessage("Tell me a joke"))
```

### Complex Conversations

For multi-turn conversations or system messages, use the full message syntax:

```go
// Full message chain with system prompt and conversation history
messages := []echo.Message{
    {Role: echo.System, Content: "You are a helpful math tutor."},
    {Role: echo.User, Content: "What is 2+2?"},
    {Role: echo.Agent, Content: "2+2 equals 4."},
    {Role: echo.User, Content: "Can you explain why?"},
}

resp, err := client.Call(ctx, messages)
```

### Available Roles

- `echo.System` - System instructions (must be first if present)
- `echo.User` - User messages
- `echo.Agent` - Assistant/model messages

## Options and Configuration

### Client Creation with Options

```go
// Set defaults at client creation time
client, _ := echo.NewClient("gemini/best", "your-api-key",
    echo.WithSystemMessage("You are a creative assistant."),
    echo.WithTemperature(0.8),
)

// Use client defaults
resp, _ := client.Call(ctx, echo.QuickMessage("Tell me a joke"))

// Override defaults for specific calls
resp, _ = client.Call(ctx, echo.QuickMessage("Write a formal email"),
    echo.WithTemperature(0.2), // More deterministic
)
```

### Per-Call Options

```go
resp, err := client.Call(ctx, echo.QuickMessage("Write a story"),
    echo.WithTemperature(0.7),
    echo.WithMaxTokens(100),
    echo.WithSystemMessage("You are a creative writer."),
)
```

### Available Options

- `WithModel(string)` - Override model for this call
- `WithTemperature(float64)` - Control randomness (0.0 - 1.0)
- `WithMaxTokens(int)` - Limit response length
- `WithSystemMessage(string)` - Set or override system prompt (overrides any system message in the message chain)

## Streaming Responses

For real-time streaming of responses, use the `StreamCall` method:

### Basic Streaming

```go
streamResp, err := client.StreamCall(ctx, echo.QuickMessage("Write a short story"))
if err != nil {
    panic(err)
}
```

### StreamCall vs Call

- **`Call`**: Returns complete response after generation finishes
- **`StreamCall`**: Returns chunks as they're generated for real-time display

Both methods support the same options

## Direct Provider Clients

For direct provider access, you can use provider-specific constructors:

```go
// OpenAI
client := echo.NewOpenAIClient("api-key", "gpt-4.1-mini")

// Anthropic
client := echo.NewAnthropicClient("api-key", "claude-opus-4-20250514")

// Gemini
client := echo.NewGeminiClient("api-key", "gemini-2.5-pro")
```

## License

MIT