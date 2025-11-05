package google

import (
	"simpleai"
	"testing"
)

func TestNewProvider(t *testing.T) {
	// Test with valid config
	config := map[string]interface{}{
		"api_key":        "test-api-key",
		"default_model":  "gemini-2.0-flash",
		"timeout":        60,
		"retry_attempts": 3,
	}

	provider, err := NewProvider(config)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	if provider == nil {
		t.Fatal("Provider is nil")
	}

	// Test provider interface implementation
	if provider.Name() != "google" {
		t.Errorf("Expected provider name 'google', got '%s'", provider.Name())
	}

	// Test ListModels
	models, err := provider.ListModels()
	if err != nil {
		t.Errorf("Failed to list models: %v", err)
	}
	if len(models) == 0 {
		t.Error("Expected at least one model")
	}

	// Test SupportedFeatures
	features := provider.SupportedFeatures()
	if !features.StructuredOutput {
		t.Error("Expected StructuredOutput to be true")
	}
	if !features.Streaming {
		t.Error("Expected Streaming to be true")
	}
	if !features.Vision {
		t.Error("Expected Vision to be true")
	}
	if features.MaxTokens == 0 {
		t.Error("Expected MaxTokens to be set")
	}
}

func TestNewProviderMissingAPIKey(t *testing.T) {
	config := map[string]interface{}{
		"default_model": "gemini-2.0-flash",
	}

	_, err := NewProvider(config)
	if err == nil {
		t.Error("Expected error when API key is missing")
	}

	llmErr, ok := err.(*simpleai.LLMError)
	if !ok {
		t.Error("Expected LLMError type")
	}
	if llmErr.Type != simpleai.ErrInvalidConfig {
		t.Errorf("Expected ErrInvalidConfig, got %v", llmErr.Type)
	}
}

func TestProviderInterface(t *testing.T) {
	config := map[string]interface{}{
		"api_key":        "test-api-key",
		"default_model":  "gemini-2.0-flash",
		"timeout":        60,
		"retry_attempts": 3,
	}

	provider, err := NewProvider(config)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Verify all interface methods exist and work
	var _ simpleai.Provider = provider

	// Test Name
	name := provider.Name()
	if name != "google" {
		t.Errorf("Expected name 'google', got '%s'", name)
	}

	// Test ListModels
	models, err := provider.ListModels()
	if err != nil {
		t.Errorf("ListModels failed: %v", err)
	}
	if len(models) == 0 {
		t.Error("Expected models to be returned")
	}

	// Test SupportedFeatures
	features := provider.SupportedFeatures()
	if len(features.SupportedRoles) == 0 {
		t.Error("Expected supported roles to be set")
	}

	// Note: IsAvailable and Chat require actual API connection, so we skip those in unit tests
}

