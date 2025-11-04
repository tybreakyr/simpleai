package simpleai

import (
	"errors"
	"fmt"
	"time"
)

// Types for interacting with LLM APIs

// Model represents an available LLM model
type Model struct {
	Name string `json:"name"`
}

// SystemPrompt represents a system-level instruction for the LLM
type SystemPrompt struct {
	Content string `json:"content"`
}

// Message represents a single message in a conversation
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatResponse represents the response from an LLM
type ChatResponse struct {
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"` // Structured output data if T was specified
}

// ChatRequest represents a chat request to an LLM
type ChatRequest struct {
	SystemPrompt SystemPrompt `json:"system_prompt"`
	Messages     []Message    `json:"messages"`
	T            any          `json:"-"` // Optional: structured output shape if desired
}

// ProviderFeatures describes the capabilities supported by an LLM provider
type ProviderFeatures struct {
	StructuredOutput bool     `json:"structured_output"` // Supports JSON mode/structured output
	Streaming        bool     `json:"streaming"`         // Supports streaming responses
	Vision           bool     `json:"vision"`            // Supports image analysis
	MaxTokens        int      `json:"max_tokens"`        // Maximum context window size
	SupportedRoles   []string `json:"supported_roles"`   // Supported message roles (system, user, assistant, etc.)
	FunctionCalling  bool     `json:"function_calling"`  // Supports function/tool calling
	Temperature      bool     `json:"temperature"`       // Supports temperature parameter
	TopP             bool     `json:"top_p"`             // Supports top-p parameter
}

// ProviderConfig holds configuration for a specific provider
type ProviderConfig struct {
	Host          string            `json:"host,omitempty"`           // Provider host URL
	APIKey        string            `json:"api_key,omitempty"`        // API key for authentication
	DefaultModel  string            `json:"default_model"`           // Default model to use
	Timeout       int               `json:"timeout"`                  // Request timeout in seconds
	RetryAttempts int               `json:"retry_attempts"`          // Maximum retry attempts
	RateLimit     int               `json:"rate_limit,omitempty"`      // Requests per minute limit
	ExtraSettings map[string]string `json:"extra_settings,omitempty"` // Provider-specific settings
}

// FactoryConfig holds the complete factory configuration
type FactoryConfig struct {
	DefaultProvider   string                    `json:"default_provider"`             // Default provider to use
	Providers         map[string]ProviderConfig `json:"providers"`                   // Provider configurations
	ModelPreferences  map[string]string         `json:"model_preferences,omitempty"`  // Task-specific model preferences
	FallbackProviders []string                  `json:"fallback_providers,omitempty"` // Provider fallback order
}

// Error types for comprehensive error handling

var (
	ErrConnectionFailed  = errors.New("connection to LLM service failed")
	ErrTimeout           = errors.New("LLM request timed out")
	ErrInvalidResponse   = errors.New("invalid response from LLM")
	ErrJSONParseFailed   = errors.New("failed to parse JSON response")
	ErrModelNotAvailable = errors.New("requested model not available")
	ErrRateLimitExceeded = errors.New("rate limit exceeded")
	ErrInvalidConfig     = errors.New("invalid configuration")
	ErrOperationFailed   = errors.New("operation failed")
)

// LLMError represents an error from LLM operations
type LLMError struct {
	Type        error
	Message     string
	Retryable   bool
	Operation   string
	RetryCount  int
	LastAttempt time.Time
	Cause       error
}

func (e *LLMError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s (operation: %s, retryable: %v, attempts: %d) - caused by: %v",
			e.Type.Error(), e.Message, e.Operation, e.Retryable, e.RetryCount, e.Cause)
	}
	return fmt.Sprintf("%s: %s (operation: %s, retryable: %v, attempts: %d)",
		e.Type.Error(), e.Message, e.Operation, e.Retryable, e.RetryCount)
}

// NewLLMError creates a new LLMError
func NewLLMError(errType error, message, operation string, retryable bool, retryCount int, cause error) *LLMError {
	return &LLMError{
		Type:        errType,
		Message:     message,
		Retryable:   retryable,
		Operation:   operation,
		RetryCount:  retryCount,
		LastAttempt: time.Now(),
		Cause:       cause,
	}
}

// IsRetryable checks if an error is retryable
func IsRetryable(err error) bool {
	if llmErr, ok := err.(*LLMError); ok {
		return llmErr.Retryable
	}
	return false
}

// ValidationError represents a data validation error
type ValidationError struct {
	Field   string
	Value   interface{}
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation error for field '%s': %s (value: %v)", e.Field, e.Message, e.Value)
}

// NewValidationError creates a new ValidationError
func NewValidationError(field string, value interface{}, message string) *ValidationError {
	return &ValidationError{
		Field:   field,
		Value:   value,
		Message: message,
	}
}

// RetryConfig holds retry configuration
type RetryConfig struct {
	MaxRetries    int
	BaseDelay     time.Duration
	MaxDelay      time.Duration
	BackoffFactor float64
}

// DefaultRetryConfig returns default retry configuration
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:    3,
		BaseDelay:     2 * time.Second,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
	}
}

