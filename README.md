# SimpleAI

A lightweight Go library for interacting with Large Language Model (LLM) providers.

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

SimpleAI supports automatic JSON extraction and parsing from LLM responses. The library includes sophisticated JSON extraction that handles:
- Markdown code blocks (```json ... ```)
- Unescaped newlines and other common JSON issues
- Multiple extraction strategies with fallbacks

### Basic Example

```go
// Define your target structure
type MathResult struct {
    Question string `json:"question"`
    Answer   int    `json:"answer"`
    Steps    []string `json:"steps"`
}

var result MathResult

// Note: Provider setup is shown in the Quick Start section above
// For this example, assume 'provider' is already initialized

response, err := provider.Chat(simpleai.ChatRequest{
    SystemPrompt: simpleai.SystemPrompt{
        Content: "You are a math tutor. Always respond with valid JSON only, no additional text.",
    },
    Messages: []simpleai.Message{
        {Role: "user", Content: "Solve 15 + 27 and show your steps"},
    },
    T: &result, // Pass pointer to target structure
})

if err != nil {
    log.Fatalf("Chat request failed: %v", err)
}

// Access the structured data
if response.Data != nil {
    // Type assert to your struct
    mathResult := response.Data.(*MathResult)
    fmt.Printf("Question: %s\n", mathResult.Question)
    fmt.Printf("Answer: %d\n", mathResult.Answer)
    fmt.Println("Steps:")
    for i, step := range mathResult.Steps {
        fmt.Printf("  %d. %s\n", i+1, step)
    }
} else {
    // Fallback to raw message if structured parsing failed
    fmt.Println("Raw response:", response.Message)
}
```

### Complex Nested Structures

```go
type Person struct {
    Name     string   `json:"name"`
    Age      int      `json:"age"`
    Email    string   `json:"email"`
    Skills   []string `json:"skills"`
    Location struct {
        City    string `json:"city"`
        Country string `json:"country"`
    } `json:"location"`
}

var person Person

// Note: Provider setup is shown in the Quick Start section above
// For this example, assume 'provider' is already initialized

response, err := provider.Chat(simpleai.ChatRequest{
    SystemPrompt: simpleai.SystemPrompt{
        Content: "Return only valid JSON. Do not include any explanatory text.",
    },
    Messages: []simpleai.Message{
        {
            Role: "user", 
            Content: "Create a sample person profile with name, age, email, 3 skills, and location",
        },
    },
    T: &person,
})

if err != nil {
    log.Fatal(err)
}

if personData, ok := response.Data.(*Person); ok {
    fmt.Printf("Name: %s\n", personData.Name)
    fmt.Printf("Age: %d\n", personData.Age)
    fmt.Printf("Location: %s, %s\n", personData.Location.City, personData.Location.Country)
    fmt.Printf("Skills: %v\n", personData.Skills)
}
```

### Error Handling for Structured Output

```go
type APIResponse struct {
    Status  string `json:"status"`
    Message string `json:"message"`
}

var apiResp APIResponse

// Note: Provider setup is shown in the Quick Start section above
// For this example, assume 'provider' is already initialized

response, err := provider.Chat(simpleai.ChatRequest{
    SystemPrompt: simpleai.SystemPrompt{
        Content: "Return JSON only.",
    },
    Messages: []simpleai.Message{
        {Role: "user", Content: "Generate a success response"},
    },
    T: &apiResp,
})

if err != nil {
    // Check if it's a JSON parsing error
    if llmErr, ok := err.(*simpleai.LLMError); ok {
        if llmErr.Type == simpleai.ErrJSONParseFailed {
            fmt.Printf("JSON parsing failed. Raw response: %s\n", response.Message)
            // You might want to retry or use the raw message
        }
    }
    log.Fatal(err)
}

// Safe type assertion
if response.Data != nil {
    if data, ok := response.Data.(*APIResponse); ok {
        fmt.Printf("Status: %s, Message: %s\n", data.Status, data.Message)
    } else {
        fmt.Println("Type assertion failed, using raw response:", response.Message)
    }
}
```

### Tips for Structured Output

1. **Clear Instructions**: Always instruct the LLM to return JSON only in your system prompt
2. **Type Safety**: Use type assertions to safely access structured data
3. **Fallback**: Always check `response.Data` and have a fallback to `response.Message`
4. **Error Handling**: The library will retry on JSON parsing errors if retryable
5. **JSON Tags**: Ensure your struct fields have proper JSON tags matching the LLM's output format

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

This project is provided as-is.

