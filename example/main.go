package main

import (
	"fmt"
	"log"
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
		log.Fatalf("Failed to load config: %v", err)
	}

	// Get default provider
	provider, err := factory.GetDefaultProvider()
	if err != nil {
		log.Fatalf("Failed to get default provider: %v", err)
	}

	// Check if provider is available
	if !provider.IsAvailable() {
		log.Fatal("Ollama provider is not available. Make sure Ollama is running on localhost:11434")
	}

	// List available models
	models, err := provider.ListModels()
	if err != nil {
		log.Fatalf("Failed to list models: %v", err)
	}

	fmt.Printf("Available models:\n")
	for _, model := range models {
		fmt.Printf("  - %s\n", model.Name)
	}

	// Send a simple chat request
	fmt.Println("\nSending chat request...")
	response, err := provider.Chat(simpleai.ChatRequest{
		SystemPrompt: simpleai.SystemPrompt{
			Content: "You are a helpful assistant.",
		},
		Messages: []simpleai.Message{
			{Role: "user", Content: "Say hello in one sentence."},
		},
	})

	if err != nil {
		log.Fatalf("Chat request failed: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Message)
}

