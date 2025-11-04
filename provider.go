package simpleai

// Provider defines the interface that all LLM providers must implement
type Provider interface {
	// Chat sends a chat request and returns a response
	Chat(request ChatRequest) (ChatResponse, error)

	// ListModels returns the available models for this provider
	ListModels() ([]Model, error)

	// Name returns the provider's name (e.g., "ollama", "openai", "anthropic")
	Name() string

	// IsAvailable checks if the provider is currently available/reachable
	IsAvailable() bool

	// SupportedFeatures returns the capabilities supported by this provider
	SupportedFeatures() ProviderFeatures
}

// ProviderConstructor is a function that creates a new provider instance
type ProviderConstructor func(config map[string]interface{}) (Provider, error)

// ProviderRegistry holds information about registered providers
type ProviderRegistry struct {
	providers map[string]ProviderConstructor
}

// NewProviderRegistry creates a new provider registry
func NewProviderRegistry() *ProviderRegistry {
	return &ProviderRegistry{
		providers: make(map[string]ProviderConstructor),
	}
}

// Register adds a new provider constructor to the registry
func (r *ProviderRegistry) Register(name string, constructor ProviderConstructor) {
	r.providers[name] = constructor
}

// Create creates a new provider instance using the registered constructor
func (r *ProviderRegistry) Create(name string, config map[string]interface{}) (Provider, error) {
	constructor, exists := r.providers[name]
	if !exists {
		return nil, NewLLMError(ErrInvalidConfig, "provider not found: "+name, "provider_creation", false, 0, nil)
	}

	return constructor(config)
}

// List returns the names of all registered providers
func (r *ProviderRegistry) List() []string {
	var names []string
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}

// Exists checks if a provider with the given name is registered
func (r *ProviderRegistry) Exists(name string) bool {
	_, exists := r.providers[name]
	return exists
}

