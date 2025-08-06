# Echo - Simple LLM Client Library

A lightweight Go library for interacting with various LLM providers with a simple, unified API.

## Supported Providers

- OpenAI
- Anthropic
- Google
- OpenRouter (via OpenAI-compatible API)

## Installation

```bash
go get github.com/mkozhukh/echo
``` 

## Quick Start

### Universal Client

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
client, _ := echo.NewClient("google/light", "")      // Uses gemini-2.5-flash
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
os.Setenv("GOOGLE_API_KEY", "your-google-key")

// API key is automatically selected based on provider
client, _ := echo.NewClient("openai/gpt-4.1", "")
```

## Message Chains

The library supports three ways to create message chains for conversations:

### 1. QuickMessage - Simple Single Messages

For basic single-message prompts:

```go
resp, _ := client.Call(ctx, echo.QuickMessage("Tell me a joke"))
```

### 2. TemplateMessage - Multi-Message Templates

For readable multi-turn conversations using a text template:

```go
messages := echo.TemplateMessage(`
@system:
You are a helpful math tutor.

@user:
What is 2+2?

@agent:
2+2 equals 4.

@user:
Can you explain why?
`)

resp, err := client.Call(ctx, messages)
```

Template format:
- `@role:` markers separate messages (system, user, agent)
- Content follows until the next marker or end of template
- Content can be on the same line: `@user: Hello there!`
- Multiline content is supported
- Whitespace is automatically trimmed

### 3. Manual Message Construction

For programmatic message building:

```go
messages := []echo.Message{
    {Role: echo.System, Content: "You are a helpful assistant."},
    {Role: echo.User, Content: "Hello"},
    {Role: echo.Agent, Content: "Hi! How can I help you today?"},
    {Role: echo.User, Content: "What's the weather like?"},
}

resp, err := client.Call(ctx, messages)
```

### Message Roles

- `echo.System` - System instructions (must be first if present, only one allowed)
- `echo.User` - User messages
- `echo.Agent` - Assistant/model messages (maps to "assistant" for OpenAI/Anthropic, "model" for Gemini)

## Options and Configuration

### Client Creation with Options

```go
// Set defaults at client creation time
client, _ := echo.NewClient("google/best", "your-api-key",
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

### Dynamic Provider Switching

The library supports switching providers on a per-call basis using `WithModel`:

```go
// Create a client with a default provider
client, _ := echo.NewClient("openai/gpt-4", "")

// Use different providers for different calls
resp1, _ := client.Call(ctx, echo.QuickMessage("Analyze this text"),
    echo.WithModel("anthropic/claude-3.5-sonnet"), // Use Anthropic for analysis
)

resp2, _ := client.Call(ctx, echo.QuickMessage("Generate an image description"),
    echo.WithModel("google/gemini-2.5-pro"), // Use Google for creative tasks
)

resp3, _ := client.Call(ctx, echo.QuickMessage("Quick calculation"),
    echo.WithModel("openai/gpt-4.1-mini"), // Use a lighter model for simple tasks
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
- `WithBaseURL(string)` - Override the API base URL (useful for custom endpoints)
- `WithEndPoint(string)` - Specify endpoint routing (primarily for OpenRouter provider selection)

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

### Using in Tests

The "mock" provider can be used for tests, it will return combined string of all incoming messages

```
client, _ := echo.NewClient("mock/any", "")
mockResp, err := client.Call(ctx, echo.QuickMessage("test"))
if err != nil {
    panic(err)
}
// outputs: `[user]: test`
```

### Using OpenRouter

OpenRouter provides access to multiple LLM providers through a single API:

```go
// Basic usage with any OpenRouter model
client, _ := echo.NewClient("openrouter/claude-3.5-sonnet", "your-openrouter-key")
```

you can specify which underlying provider infrastructure to use:

```go
// Specify provider routing with @ syntax in model name
client, _ := echo.NewClient("openrouter/claude-3.5-sonnet@aws", "your-openrouter-key")

// Multiple providers for fallback (comma-separated)
client, _ := echo.NewClient("openrouter/gpt-4@azure,openai", "your-openrouter-key")
```

## License

MIT