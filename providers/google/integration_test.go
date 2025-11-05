package google

import (
	"simpleai"
	"testing"
)

func TestProviderFactoryIntegration(t *testing.T) {
	// Test that provider can be registered with factory
	factory := simpleai.NewLLMFactory()

	// Register Google provider
	factory.RegisterProvider("google", NewProvider)

	// Verify provider is registered
	providers := factory.ListProviders()
	found := false
	for _, name := range providers {
		if name == "google" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Google provider not found in registered providers")
	}

	// Test creating provider from config
	config := simpleai.FactoryConfig{
		DefaultProvider: "google",
		Providers: map[string]simpleai.ProviderConfig{
			"google": {
				APIKey:        "test-api-key",
				DefaultModel:  "gemini-2.0-flash",
				Timeout:       60,
				RetryAttempts: 3,
			},
		},
	}

	if err := factory.LoadConfig(config); err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Verify provider can be created
	provider, err := factory.CreateProviderFromConfig("google")
	if err != nil {
		t.Fatalf("Failed to create provider from config: %v", err)
	}

	if provider == nil {
		t.Fatal("Provider is nil")
	}

	// Verify provider implements interface
	if provider.Name() != "google" {
		t.Errorf("Expected name 'google', got '%s'", provider.Name())
	}
}

