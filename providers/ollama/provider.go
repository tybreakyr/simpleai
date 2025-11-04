package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"simpleai"
	ollamaclient "simpleai/ollama"
	"time"

	"github.com/ollama/ollama/api"
)

// Provider implements the simpleai.Provider interface for Ollama
type Provider struct {
	client       *ollamaclient.Client
	host         string
	defaultModel string
	timeout      int
	retryConfig  *simpleai.RetryConfig
	ollamaClient *api.Client // Direct access for provider-specific operations
}

// NewProvider creates a new Ollama provider instance
func NewProvider(config map[string]interface{}) (simpleai.Provider, error) {
	// Extract configuration values with defaults
	host := "http://localhost:11434"
	if h, ok := config["host"].(string); ok && h != "" {
		host = h
	}

	defaultModel := "llama3.1:latest"
	if m, ok := config["default_model"].(string); ok && m != "" {
		defaultModel = m
	}

	timeout := 60
	if t, ok := config["timeout"].(int); ok && t > 0 {
		timeout = t
	}

	retryAttempts := 3
	if r, ok := config["retry_attempts"].(int); ok && r >= 0 {
		retryAttempts = r
	}

	// Create retry configuration
	retryConfig := &simpleai.RetryConfig{
		MaxRetries:    retryAttempts,
		BaseDelay:     2 * time.Second,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
	}

	// Parse host URL for ollama client
	hostURL, err := url.Parse(host)
	if err != nil {
		return nil, simpleai.NewLLMError(simpleai.ErrInvalidConfig,
			fmt.Sprintf("invalid host URL: %s", host),
			"provider_creation", false, 0, err)
	}

	// Create the direct ollama API client for provider operations
	ollamaClient := api.NewClient(hostURL, http.DefaultClient)

	// Create the wrapped ollama client with default model
	model := simpleai.Model{Name: defaultModel}
	wrappedClient := ollamaclient.NewClient(model)

	return &Provider{
		client:       wrappedClient,
		host:         host,
		defaultModel: defaultModel,
		timeout:      timeout,
		retryConfig:  retryConfig,
		ollamaClient: ollamaClient,
	}, nil
}

// Chat sends a chat request and returns a response
func (p *Provider) Chat(request simpleai.ChatRequest) (simpleai.ChatResponse, error) {
	// Use the wrapped ollama client's ChatWithRetry method
	return p.client.ChatWithRetry(request, p.retryConfig)
}

// ListModels returns the available models for this provider
func (p *Provider) ListModels() ([]simpleai.Model, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(p.timeout)*time.Second)
	defer cancel()

	// Use the direct ollama API client to list models
	listResp, err := p.ollamaClient.List(ctx)
	if err != nil {
		return nil, simpleai.NewLLMError(simpleai.ErrOperationFailed,
			"failed to list models from Ollama",
			"list_models", true, 0, err)
	}

	// Convert ollama models to simpleai.Model format
	models := make([]simpleai.Model, len(listResp.Models))
	for i, model := range listResp.Models {
		models[i] = simpleai.Model{Name: model.Name}
	}

	return models, nil
}

// Name returns the provider's name
func (p *Provider) Name() string {
	return "ollama"
}

// IsAvailable checks if the provider is currently available/reachable
func (p *Provider) IsAvailable() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try to list models as a simple health check
	_, err := p.ollamaClient.List(ctx)
	return err == nil
}

// SupportedFeatures returns the capabilities supported by this provider
func (p *Provider) SupportedFeatures() simpleai.ProviderFeatures {
	return simpleai.ProviderFeatures{
		StructuredOutput: true,  // Ollama supports structured JSON output via prompting
		Streaming:        true,  // Ollama supports streaming responses
		Vision:           false, // Basic Ollama doesn't support vision (depends on model)
		MaxTokens:        4096,  // Default context window (varies by model)
		SupportedRoles:   []string{"system", "user", "assistant"},
		FunctionCalling:  false, // Ollama doesn't natively support function calling
		Temperature:      true,  // Ollama supports temperature parameter
		TopP:             true,  // Ollama supports top-p parameter
	}
}

// GetDefaultModel returns the default model for this provider
func (p *Provider) GetDefaultModel() string {
	return p.defaultModel
}

// CreateClientWithModel creates a new ollama client instance with a specific model
// This method allows the provider to create model-specific clients when needed
func (p *Provider) CreateClientWithModel(modelName string) *ollamaclient.Client {
	model := simpleai.Model{Name: modelName}
	return ollamaclient.NewClient(model)
}

// GetRetryConfig returns the retry configuration for this provider
func (p *Provider) GetRetryConfig() *simpleai.RetryConfig {
	return p.retryConfig
}

// UpdateRetryConfig allows updating the retry configuration
func (p *Provider) UpdateRetryConfig(config *simpleai.RetryConfig) {
	if config != nil {
		p.retryConfig = config
	}
}

