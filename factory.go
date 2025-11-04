package simpleai

import (
	"fmt"
	"sync"
)

// Factory manages provider creation and model discovery
type Factory interface {
	// CreateProvider creates a provider instance with the given configuration
	CreateProvider(providerName string, config map[string]interface{}) (Provider, error)

	// CreateProviderFromConfig creates a provider using the factory's stored configuration
	CreateProviderFromConfig(providerName string) (Provider, error)

	// ListProviders returns the names of all registered providers
	ListProviders() []string

	// ListModels returns available models for a specific provider
	ListModels(providerName string) ([]Model, error)

	// GetDefaultProvider returns the configured default provider
	GetDefaultProvider() (Provider, error)

	// SetDefaultProvider sets the default provider name
	SetDefaultProvider(providerName string) error

	// GetDefaultProviderName returns the name of the default provider
	GetDefaultProviderName() string

	// IsProviderAvailable checks if a provider is registered and available
	IsProviderAvailable(providerName string) bool

	// GetProviderFeatures returns the features supported by a provider
	GetProviderFeatures(providerName string) (ProviderFeatures, error)

	// LoadConfig loads factory configuration from a FactoryConfig struct
	LoadConfig(config FactoryConfig) error

	// GetConfig returns the current factory configuration
	GetConfig() FactoryConfig
}

// LLMFactory is the concrete implementation of the Factory interface
type LLMFactory struct {
	mu            sync.RWMutex
	registry      *ProviderRegistry
	config        FactoryConfig
	providerCache map[string]Provider // Cache for created providers
	modelCache    map[string][]Model  // Cache for model lists
}

// NewLLMFactory creates a new LLM factory with default configuration
func NewLLMFactory() *LLMFactory {
	return &LLMFactory{
		registry:      NewProviderRegistry(),
		config:        FactoryConfig{},
		providerCache: make(map[string]Provider),
		modelCache:    make(map[string][]Model),
	}
}

// RegisterProvider registers a new provider constructor with the factory
func (f *LLMFactory) RegisterProvider(name string, constructor ProviderConstructor) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.registry.Register(name, constructor)
}

// CreateProvider creates a provider instance with the given configuration
func (f *LLMFactory) CreateProvider(providerName string, config map[string]interface{}) (Provider, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.createProviderUnsafe(providerName, config)
}

// createProviderUnsafe creates a provider without acquiring locks (internal use only)
func (f *LLMFactory) createProviderUnsafe(providerName string, config map[string]interface{}) (Provider, error) {
	// Check if provider is cached
	if provider, exists := f.providerCache[providerName]; exists {
		return provider, nil
	}

	// Create new provider instance
	provider, err := f.registry.Create(providerName, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider %s: %w", providerName, err)
	}

	// Cache the provider for reuse
	f.providerCache[providerName] = provider

	return provider, nil
}

// CreateProviderFromConfig creates a provider using the factory's stored configuration
func (f *LLMFactory) CreateProviderFromConfig(providerName string) (Provider, error) {
	f.mu.RLock()
	providerConfig, exists := f.config.Providers[providerName]
	f.mu.RUnlock()

	if !exists {
		return nil, NewLLMError(ErrInvalidConfig,
			fmt.Sprintf("no configuration found for provider: %s", providerName),
			"provider_creation", false, 0, nil)
	}

	// Convert ProviderConfig to map[string]interface{}
	configMap := map[string]interface{}{
		"host":           providerConfig.Host,
		"api_key":        providerConfig.APIKey,
		"default_model":  providerConfig.DefaultModel,
		"timeout":        providerConfig.Timeout,
		"retry_attempts": providerConfig.RetryAttempts,
		"rate_limit":     providerConfig.RateLimit,
	}

	// Add extra settings
	for key, value := range providerConfig.ExtraSettings {
		configMap[key] = value
	}

	return f.CreateProvider(providerName, configMap)
}

// ListProviders returns the names of all registered providers
func (f *LLMFactory) ListProviders() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.registry.List()
}

// ListModels returns available models for a specific provider
func (f *LLMFactory) ListModels(providerName string) ([]Model, error) {
	f.mu.RLock()
	// Check cache first
	if models, exists := f.modelCache[providerName]; exists {
		f.mu.RUnlock()
		return models, nil
	}
	f.mu.RUnlock()

	// Create provider to list models
	provider, err := f.CreateProviderFromConfig(providerName)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider for model listing: %w", err)
	}

	models, err := provider.ListModels()
	if err != nil {
		return nil, fmt.Errorf("failed to list models for provider %s: %w", providerName, err)
	}

	// Cache the results
	f.mu.Lock()
	f.modelCache[providerName] = models
	f.mu.Unlock()

	return models, nil
}

// GetDefaultProvider returns the configured default provider
func (f *LLMFactory) GetDefaultProvider() (Provider, error) {
	f.mu.RLock()
	defaultProviderName := f.config.DefaultProvider
	f.mu.RUnlock()

	if defaultProviderName == "" {
		return nil, NewLLMError(ErrInvalidConfig,
			"no default provider configured",
			"get_default_provider", false, 0, nil)
	}

	return f.CreateProviderFromConfig(defaultProviderName)
}

// SetDefaultProvider sets the default provider name
func (f *LLMFactory) SetDefaultProvider(providerName string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.registry.Exists(providerName) {
		return NewLLMError(ErrInvalidConfig,
			fmt.Sprintf("provider not registered: %s", providerName),
			"set_default_provider", false, 0, nil)
	}

	f.config.DefaultProvider = providerName
	return nil
}

// GetDefaultProviderName returns the name of the default provider
func (f *LLMFactory) GetDefaultProviderName() string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.config.DefaultProvider
}

// IsProviderAvailable checks if a provider is registered and available
func (f *LLMFactory) IsProviderAvailable(providerName string) bool {
	f.mu.RLock()
	registryExists := f.registry.Exists(providerName)
	f.mu.RUnlock()

	if !registryExists {
		return false
	}

	// Try to create the provider to check availability
	// Don't hold any locks during provider creation to avoid deadlock
	provider, err := f.CreateProviderFromConfig(providerName)
	if err != nil {
		return false
	}

	return provider.IsAvailable()
}

// GetProviderFeatures returns the features supported by a provider
func (f *LLMFactory) GetProviderFeatures(providerName string) (ProviderFeatures, error) {
	provider, err := f.CreateProviderFromConfig(providerName)
	if err != nil {
		return ProviderFeatures{}, fmt.Errorf("failed to get provider features: %w", err)
	}

	return provider.SupportedFeatures(), nil
}

// LoadConfig loads factory configuration from a FactoryConfig struct
func (f *LLMFactory) LoadConfig(config FactoryConfig) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Validate configuration
	if config.DefaultProvider == "" {
		return NewLLMError(ErrInvalidConfig,
			"default provider must be specified",
			"load_config", false, 0, nil)
	}

	if len(config.Providers) == 0 {
		return NewLLMError(ErrInvalidConfig,
			"at least one provider must be configured",
			"load_config", false, 0, nil)
	}

	// Check if default provider exists in configuration
	if _, exists := config.Providers[config.DefaultProvider]; !exists {
		return NewLLMError(ErrInvalidConfig,
			fmt.Sprintf("default provider %s not found in provider configurations", config.DefaultProvider),
			"load_config", false, 0, nil)
	}

	f.config = config

	// Clear caches when config changes
	f.providerCache = make(map[string]Provider)
	f.modelCache = make(map[string][]Model)

	return nil
}

// GetConfig returns the current factory configuration
func (f *LLMFactory) GetConfig() FactoryConfig {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.config
}

// ClearModelCache clears the cached model lists (useful for refreshing)
func (f *LLMFactory) ClearModelCache() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.modelCache = make(map[string][]Model)
}

// ClearProviderCache clears the cached provider instances
func (f *LLMFactory) ClearProviderCache() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.providerCache = make(map[string]Provider)
}

