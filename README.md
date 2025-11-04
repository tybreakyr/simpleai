# SimpleAI

A lightweight Go library for interacting with Large Language Model (LLM) providers. This is a stripped-down version focused solely on LLM interaction, without story-specific functionality.

## Features

- **Provider Abstraction**: Clean interface for different LLM providers
- **Factory Pattern**: Easy provider management and configuration
- **Ollama Support**: Built-in support for Ollama
- **Structured Output**: Automatic JSON extraction and parsing
- **Retry Logic**: Configurable retry with exponential backoff
- **Error Handling**: Comprehensive error types and classification
- **Configuration**: Flexible configuration system

## Installation

```bash
go get simpleai
```

## Quick Start

```go
package main

import (
    "fmt"
    "simpleai"
    ollamaprovider "simpleai/providers/ollama"
)

func main() {
    // Create factory
    factory := simpleai.NewLLMFactory()
    
    // Register Ollama provider
    factory.RegisterProvider("ollama", ollamaprovider.NewProvider)
    
    // Load configuration
    config := simpleai.FactoryConfig{
        DefaultProvider: "ollama",
        Providers: map[string]simpleai.ProviderConfig{
            "ollama": {
                Host:          "http://localhost:11434",
                DefaultModel:  "llama3.1:latest",
                Timeout:       60,
                RetryAttempts: 3,
            },
        },
    }
    
    if err := factory.LoadConfig(config); err != nil {
        panic(err)
    }
    
    // Get default provider
    provider, err := factory.GetDefaultProvider()
    if err != nil {
        panic(err)
    }
    
    // Send chat request
    response, err := provider.Chat(simpleai.ChatRequest{
        SystemPrompt: simpleai.SystemPrompt{
            Content: "You are a helpful assistant.",
        },
        Messages: []simpleai.Message{
            {Role: "user", Content: "Hello!"},
        },
    })
    
    if err != nil {
        panic(err)
    }
    
    fmt.Println(response.Message)
}
```

## Configuration

### Factory Configuration

```go
config := simpleai.FactoryConfig{
    DefaultProvider: "ollama",
    Providers: map[string]simpleai.ProviderConfig{
        "ollama": {
            Host:          "http://localhost:11434",
            APIKey:        "", // Not needed for Ollama
            DefaultModel:  "llama3.1:latest",
            Timeout:       60,
            RetryAttempts: 3,
            RateLimit:     0, // Optional
            ExtraSettings: map[string]string{}, // Optional
        },
    },
    ModelPreferences: map[string]string{
        "default": "llama3.1:latest",
    },
    FallbackProviders: []string{"ollama"},
}
```

### Provider Configuration

Each provider can be configured with:
- `host`: Provider API endpoint
- `api_key`: API key (if required)
- `default_model`: Default model to use
- `timeout`: Request timeout in seconds
- `retry_attempts`: Maximum retry attempts
- `rate_limit`: Rate limit (requests per minute)
- `extra_settings`: Provider-specific settings

## Structured Output

You can request structured JSON output by providing a target type:

```go
type Response struct {
    Answer string `json:"answer"`
    Score  int    `json:"score"`
}

var target Response

response, err := provider.Chat(simpleai.ChatRequest{
    SystemPrompt: simpleai.SystemPrompt{
        Content: "Return JSON only.",
    },
    Messages: []simpleai.Message{
        {Role: "user", Content: "What is 2+2?"},
    },
    T: &target, // Target structure
})

// Access structured data
if response.Data != nil {
    resp := response.Data.(*Response)
    fmt.Println(resp.Answer)
}
```

## Error Handling

The library provides comprehensive error handling:

```go
response, err := provider.Chat(request)
if err != nil {
    if simpleai.IsRetryable(err) {
        // Error is retryable
    }
    
    if llmErr, ok := err.(*simpleai.LLMError); ok {
        fmt.Printf("Operation: %s\n", llmErr.Operation)
        fmt.Printf("Retryable: %v\n", llmErr.Retryable)
        fmt.Printf("Retry Count: %d\n", llmErr.RetryCount)
    }
}
```

### Error Types

- `ErrConnectionFailed`: Connection to LLM service failed
- `ErrTimeout`: Request timed out
- `ErrInvalidResponse`: Invalid response from LLM
- `ErrJSONParseFailed`: Failed to parse JSON response
- `ErrModelNotAvailable`: Requested model not available
- `ErrRateLimitExceeded`: Rate limit exceeded
- `ErrInvalidConfig`: Invalid configuration
- `ErrOperationFailed`: General operation failure

## Provider Interface

All providers implement the `Provider` interface:

```go
type Provider interface {
    Chat(request ChatRequest) (ChatResponse, error)
    ListModels() ([]Model, error)
    Name() string
    IsAvailable() bool
    SupportedFeatures() ProviderFeatures
}
```

## Adding New Providers

To add a new provider:

1. Implement the `Provider` interface
2. Create a constructor function: `func NewProvider(config map[string]interface{}) (Provider, error)`
3. Register with the factory: `factory.RegisterProvider("name", NewProvider)`
4. Configure in `FactoryConfig`

## Retry Configuration

Default retry configuration:
- Max retries: 3
- Base delay: 2 seconds
- Max delay: 30 seconds
- Backoff factor: 2.0

You can customize retry behavior by modifying the provider's retry config.

## JSON Extraction

The library includes sophisticated JSON extraction from LLM responses:
- Handles markdown code blocks (```json)
- Repairs common JSON issues (unescaped newlines)
- Uses brace counting for complex responses
- Aggressive cleanup strategies

## License

This project is provided as-is for use in the storyteller project.

