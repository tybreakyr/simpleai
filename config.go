package simpleai

import (
	"encoding/json"
	"fmt"
)

// Config holds basic configuration (legacy format for migration)
type Config struct {
	OllamaHost     string            `json:"ollama_host"`
	DefaultModel   string            `json:"default_model"`
	DefaultPrompts map[string]string `json:"default_prompts,omitempty"`
}

// ConfigMigrator handles migration from old config format to new factory config format
type ConfigMigrator struct{}

// NewConfigMigrator creates a new configuration migrator
func NewConfigMigrator() *ConfigMigrator {
	return &ConfigMigrator{}
}

// MigrateFromOldConfig converts the old config format to the new FactoryConfig format
func (m *ConfigMigrator) MigrateFromOldConfig(oldConfig Config) FactoryConfig {
	// Create the new factory config
	factoryConfig := FactoryConfig{
		DefaultProvider: "ollama",
		Providers: map[string]ProviderConfig{
			"ollama": {
				Host:          oldConfig.OllamaHost,
				DefaultModel:  oldConfig.DefaultModel,
				Timeout:       60, // Default timeout
				RetryAttempts: 3,  // Default retry attempts
			},
		},
		ModelPreferences: map[string]string{
			"default": oldConfig.DefaultModel,
		},
	}

	return factoryConfig
}

// MigrateFromJSON converts JSON config data to FactoryConfig
func (m *ConfigMigrator) MigrateFromJSON(jsonData []byte) (FactoryConfig, error) {
	// Try to unmarshal as new format first
	var factoryConfig FactoryConfig
	if err := json.Unmarshal(jsonData, &factoryConfig); err == nil && factoryConfig.DefaultProvider != "" {
		return factoryConfig, nil
	}

	// Try to unmarshal as old format
	var oldConfig Config
	if err := json.Unmarshal(jsonData, &oldConfig); err != nil {
		return FactoryConfig{}, fmt.Errorf("failed to parse config in either old or new format: %w", err)
	}

	// Migrate from old format
	return m.MigrateFromOldConfig(oldConfig), nil
}

// IsOldConfigFormat checks if the JSON data represents the old config format
func (m *ConfigMigrator) IsOldConfigFormat(jsonData []byte) bool {
	var oldConfig Config
	if err := json.Unmarshal(jsonData, &oldConfig); err != nil {
		return false
	}

	// Check for old format indicators
	return oldConfig.OllamaHost != "" || oldConfig.DefaultModel != ""
}

// GenerateDefaultFactoryConfig creates a default factory configuration
func GenerateDefaultFactoryConfig() FactoryConfig {
	return FactoryConfig{
		DefaultProvider: "ollama",
		Providers: map[string]ProviderConfig{
			"ollama": {
				Host:          "http://localhost:11434",
				DefaultModel:  "llama3.1:latest",
				Timeout:       60,
				RetryAttempts: 3,
			},
		},
		ModelPreferences: map[string]string{
			"default": "llama3.1:latest",
		},
		FallbackProviders: []string{"ollama"},
	}
}

// ValidateFactoryConfig validates a factory configuration
func ValidateFactoryConfig(config FactoryConfig) error {
	if config.DefaultProvider == "" {
		return NewLLMError(ErrInvalidConfig, "default provider must be specified", "config_validation", false, 0, nil)
	}

	if len(config.Providers) == 0 {
		return NewLLMError(ErrInvalidConfig, "at least one provider must be configured", "config_validation", false, 0, nil)
	}

	// Check if default provider exists in configuration
	if _, exists := config.Providers[config.DefaultProvider]; !exists {
		return NewLLMError(ErrInvalidConfig,
			fmt.Sprintf("default provider %s not found in provider configurations", config.DefaultProvider),
			"config_validation", false, 0, nil)
	}

	// Validate each provider configuration
	for name, providerConfig := range config.Providers {
		if providerConfig.DefaultModel == "" {
			return NewLLMError(ErrInvalidConfig,
				fmt.Sprintf("provider %s must have a default model specified", name),
				"config_validation", false, 0, nil)
		}

		if providerConfig.Timeout <= 0 {
			return NewLLMError(ErrInvalidConfig,
				fmt.Sprintf("provider %s must have a positive timeout value", name),
				"config_validation", false, 0, nil)
		}

		if providerConfig.RetryAttempts < 0 {
			return NewLLMError(ErrInvalidConfig,
				fmt.Sprintf("provider %s cannot have negative retry attempts", name),
				"config_validation", false, 0, nil)
		}
	}

	return nil
}

