# Echo - Simple LLM Client Library

A lightweight Go library for interacting with various LLM providers with a simple, unified API.

## Supported Providers

- OpenAI
- Google Gemini
- Anthropic (Claude)


## Installation

```bash
go get github.com/mkozhukh/echo
``` 

## Features

### Simple Usage (Recommended)

```go
package main

import (
    "context"
    "fmt"
    "github.com/mkozhukh/echo"
)

func main() {
    ctx := context.Background()

    // just API key
    geminiClient := echo.NewGemini25Pro("your-gemini-api-key")
    resp, err := geminiClient.Call(ctx, "Hello, how are you?")
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Text)

    // Anthropic example
    claudeClient := echo.NewClaude3Sonnet("your-anthropic-api-key")
    resp, err = claudeClient.Call(ctx, "Tell me a joke")
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Text)

    // OpenAI GPT-4.1 example
    gptClient := echo.NewGPT41("your-openai-api-key")
    resp, err = gptClient.Call(ctx, "Explain quantum computing")
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Text)
}
```

### Client Creation with Options

```go
// Set defaults at client creation time
creativeClient := echo.NewGemini25Pro("your-api-key",
    echo.WithSystemMessage("You are a creative assistant."),
    echo.WithTemperature(0.8)
)

// Use client defaults
resp, err := creativeClient.Call(ctx, "Tell me a joke")

// Override defaults for specific calls
resp, err = creativeClient.Call(ctx, "Write a formal email",
    echo.WithTemperature(0.2), // More deterministic
)
```

### Custom Model with Options

```go
// Gemini with custom model
client := echo.NewGeminiClient("your-api-key", "gemini-2.0-flash",
    echo.WithClientSystemMessage("You are a helpful coding assistant."),
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

## API

### OpenAI
- `NewOpenAIClient(apiKey, model, opts...)`
- `NewGPT41(apiKey, opts...)` - GPT-4.1
- `NewGPT41Mini(apiKey, opts...)` - GPT-4.1 Mini
- `NewGPT41Nano(apiKey, opts...)` - GPT-4.1 Nano

### Gemini
- `NewGeminiClient(apiKey, model, opts...)`
- `NewGemini25Pro(apiKey, opts...)` - Gemini Pro
- `NewGemini25Flash(apiKey, opts...)` - Gemini Flash

### Anthropic
- `NewAnthropicClient(apiKey, model, opts...)`
- `NewClaude4Opus(apiKey, opts...)` - Claude 3 Opus
- `NewClaude4Sonnet(apiKey, opts...)` - Claude 3.5 Sonnet
- `NewClaude35Haiku(apiKey, opts...)` - Claude 3.5 Haiku


### Option Functions

- `WithModel(string)` - Override model for this call
- `WithTemperature(float64)` - Override temperature for this call
- `WithMaxTokens(int)` - Override max tokens for this call
- `WithSystemMessage(string)` - Override system message for this call


## CLI Tool

A simple command-line utility `ec` is included for quick testing. It's not the main feature of this library.

### Installation

```bash
go install github.com/mkozhukh/echo/cmd/ec@latest
```

### Usage

```bash
# With specific model and key
ec --model gemini/gemini-2.5-flash --key YOUR_KEY "Hello world"

# Using environment variables
export ECHO_MODEL=anthropic/claude-3-sonnet
export ECHO_KEY=your-api-key
ec "Explain quantum computing"

# With custom system prompt
ec --prompt "You are a helpful coding assistant" "Write a hello world in Go"
```

## License

MIT