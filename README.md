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
    
    resp, err := client.Call(ctx, "Hello, how are you?")
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

## Options and Configuration

### Client Creation with Options

```go
// Set defaults at client creation time
client, _ := echo.NewClient("gemini/best", "your-api-key",
    echo.WithSystemMessage("You are a creative assistant."),
    echo.WithTemperature(0.8),
)

// Use client defaults
resp, _ := client.Call(ctx, "Tell me a joke")

// Override defaults for specific calls
resp, _ = client.Call(ctx, "Write a formal email",
    echo.WithTemperature(0.2), // More deterministic
)
```

### Per-Call Options

```go
resp, err := client.Call(ctx, "Write a story",
    echo.WithTemperature(0.7),
    echo.WithMaxTokens(100),
    echo.WithSystemMessage("You are a creative writer."),
)
```

### Available Options

- `WithModel(string)` - Override model for this call
- `WithTemperature(float64)` - Control randomness (0.0 - 1.0)
- `WithMaxTokens(int)` - Limit response length
- `WithSystemMessage(string)` - Set system prompt

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