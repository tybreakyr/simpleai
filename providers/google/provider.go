package google

import (
	"context"
	"encoding/json"
	"fmt"
	"simpleai"
	"strings"
	"time"

	"google.golang.org/genai"
)

// Provider implements the simpleai.Provider interface for Google Gemini
type Provider struct {
	client       *genai.Client
	defaultModel string
	timeout      int
	retryConfig  *simpleai.RetryConfig
}

// NewProvider creates a new Google provider instance
func NewProvider(config map[string]interface{}) (simpleai.Provider, error) {
	// Extract API key (required)
	apiKey := ""
	if key, ok := config["api_key"].(string); ok && key != "" {
		apiKey = key
	} else {
		return nil, simpleai.NewLLMError(simpleai.ErrInvalidConfig,
			"API key is required for Google provider",
			"provider_creation", false, 0, nil)
	}

	// Extract default model
	defaultModel := "gemini-2.0-flash"
	if m, ok := config["default_model"].(string); ok && m != "" {
		defaultModel = m
	}

	// Extract timeout
	timeout := 60
	if t, ok := config["timeout"].(int); ok && t > 0 {
		timeout = t
	}

	// Extract retry attempts
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

	// Create context for client initialization
	ctx := context.Background()

	// Create Google Gen AI client
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, simpleai.NewLLMError(simpleai.ErrConnectionFailed,
			"failed to create Google Gen AI client",
			"provider_creation", false, 0, err)
	}

	return &Provider{
		client:       client,
		defaultModel: defaultModel,
		timeout:      timeout,
		retryConfig:  retryConfig,
	}, nil
}

// Chat sends a chat request and returns a response
func (p *Provider) Chat(request simpleai.ChatRequest) (simpleai.ChatResponse, error) {
	return p.ChatWithRetry(request, p.retryConfig)
}

// ChatWithRetry executes a chat request with retry logic
func (p *Provider) ChatWithRetry(request simpleai.ChatRequest, retryConfig *simpleai.RetryConfig) (simpleai.ChatResponse, error) {
	var lastErr error
	var finalResponse string
	var structuredData any
	var targetType interface{} = request.T

	for attempt := 0; attempt <= retryConfig.MaxRetries; attempt++ {
		// Calculate delay for retry (exponential backoff)
		if attempt > 0 {
			delay := time.Duration(float64(retryConfig.BaseDelay) * pow(retryConfig.BackoffFactor, attempt-1))
			if delay > retryConfig.MaxDelay {
				delay = retryConfig.MaxDelay
			}
			time.Sleep(delay)
		}

		// Reset response for each attempt
		finalResponse = ""
		structuredData = nil

		// Create context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(p.timeout)*time.Second)

		// Convert messages to genai.Content format
		contents := convertMessages(request.Messages, request.SystemPrompt)

		// Create generation config with system instruction if provided
		var genConfig *genai.GenerateContentConfig
		if request.SystemPrompt.Content != "" {
			// Create Content for system instruction
			systemContent := genai.NewContentFromText(request.SystemPrompt.Content, genai.RoleUser)
			genConfig = &genai.GenerateContentConfig{
				SystemInstruction: systemContent,
			}
		}

		// Call GenerateContent
		resp, err := p.client.Models.GenerateContent(ctx, p.defaultModel, contents, genConfig)
		cancel()

		if err != nil {
			lastErr = p.classifyError(err, attempt, "chat")
			if attempt < retryConfig.MaxRetries && simpleai.IsRetryable(lastErr) {
				continue // Retry
			}
			break // No more retries or non-retryable error
		}

		// Extract text from response
		if resp != nil {
			finalResponse = resp.Text()
		}

		if finalResponse == "" {
			lastErr = simpleai.NewLLMError(simpleai.ErrInvalidResponse,
				"empty response from Google API",
				"chat", true, attempt, nil)
			if attempt < retryConfig.MaxRetries {
				continue
			}
			break
		}

		// Handle structured data parsing after complete response
		if targetType != nil {
			// Try to extract JSON from the response
			jsonStr := extractJSON(finalResponse)
			if jsonStr == "" {
				responsePreview := finalResponse
				if len(responsePreview) > 200 {
					responsePreview = responsePreview[:200] + "..."
				}
				lastErr = simpleai.NewLLMError(simpleai.ErrJSONParseFailed,
					"no valid JSON found in response", "chat", true, attempt,
					fmt.Errorf("extractJSON failed - response preview: %s", responsePreview))
				if attempt < retryConfig.MaxRetries {
					continue // Retry for JSON parsing issues
				}
				break
			}

			if err := json.Unmarshal([]byte(jsonStr), targetType); err != nil {
				jsonPreview := jsonStr
				if len(jsonPreview) > 200 {
					jsonPreview = jsonPreview[:200] + "..."
				}
				lastErr = simpleai.NewLLMError(simpleai.ErrJSONParseFailed,
					"failed to parse JSON response", "chat", true, attempt,
					fmt.Errorf("unmarshal error: %v - extracted JSON preview: %s", err, jsonPreview))
				if attempt < retryConfig.MaxRetries {
					continue // Retry for JSON parsing issues
				}
				break
			}
			structuredData = targetType
		}

		// Success - return the response
		return simpleai.ChatResponse{Message: finalResponse, Data: structuredData}, nil
	}

	// All retries exhausted
	return simpleai.ChatResponse{}, lastErr
}

// convertMessages converts simpleai messages to genai.Content format
func convertMessages(messages []simpleai.Message, systemPrompt simpleai.SystemPrompt) []*genai.Content {
	contents := make([]*genai.Content, 0, len(messages))

	// Add system prompt as first message if provided and not already in messages
	hasSystemInMessages := false
	for _, msg := range messages {
		if msg.Role == "system" {
			hasSystemInMessages = true
			break
		}
	}

	// Convert messages to contents
	for _, msg := range messages {
		// Skip system messages if we're handling it via SystemInstruction
		if msg.Role == "system" && systemPrompt.Content != "" && !hasSystemInMessages {
			continue
		}

		// Map role to genai role
		var role genai.Role
		switch msg.Role {
		case "user":
			role = genai.RoleUser
		case "assistant":
			role = genai.RoleModel
		case "system":
			role = genai.RoleUser // System messages are treated as user messages in content
		default:
			role = genai.RoleUser // Default to user role
		}

		// Use NewContentFromText helper
		content := genai.NewContentFromText(msg.Content, role)
		contents = append(contents, content)
	}

	return contents
}

// ListModels returns the available models for this provider
func (p *Provider) ListModels() ([]simpleai.Model, error) {
	// Google Gemini API doesn't have a direct list models endpoint
	// Return common Gemini models
	models := []simpleai.Model{
		{Name: "gemini-2.0-flash"},
		{Name: "gemini-2.0-flash-exp"},
		{Name: "gemini-1.5-pro"},
		{Name: "gemini-1.5-flash"},
		{Name: "gemini-1.5-pro-latest"},
		{Name: "gemini-1.5-flash-latest"},
	}

	return models, nil
}

// Name returns the provider's name
func (p *Provider) Name() string {
	return "google"
}

// IsAvailable checks if the provider is currently available/reachable
func (p *Provider) IsAvailable() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try a simple health check by attempting to generate content with a minimal prompt
	testContent := genai.NewContentFromText("test", genai.RoleUser)
	_, err := p.client.Models.GenerateContent(ctx, p.defaultModel, []*genai.Content{testContent}, nil)

	return err == nil
}

// SupportedFeatures returns the capabilities supported by this provider
func (p *Provider) SupportedFeatures() simpleai.ProviderFeatures {
	return simpleai.ProviderFeatures{
		StructuredOutput: true,  // Gemini supports JSON mode
		Streaming:        true,  // Gemini supports streaming responses
		Vision:           true,  // Gemini supports image inputs
		MaxTokens:        32768, // Gemini 2.0 Flash context window
		SupportedRoles:   []string{"system", "user", "assistant"},
		FunctionCalling:  true, // Gemini supports function calling
		Temperature:      true, // Gemini supports temperature parameter
		TopP:             true, // Gemini supports top-p parameter
	}
}

// classifyError classifies errors and determines if they are retryable
func (p *Provider) classifyError(err error, attempt int, operation string) *simpleai.LLMError {
	errStr := err.Error()

	// Check for specific error types
	if strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "no such host") ||
		strings.Contains(errStr, "network is unreachable") {
		return simpleai.NewLLMError(simpleai.ErrConnectionFailed,
			"connection to Google API failed", operation, true, attempt, err)
	}

	if strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "context deadline exceeded") ||
		strings.Contains(errStr, "deadline exceeded") {
		return simpleai.NewLLMError(simpleai.ErrTimeout,
			"request timed out", operation, true, attempt, err)
	}

	if strings.Contains(errStr, "authentication") ||
		strings.Contains(errStr, "unauthorized") ||
		strings.Contains(errStr, "API key") ||
		strings.Contains(errStr, "invalid api key") {
		return simpleai.NewLLMError(simpleai.ErrInvalidConfig,
			"authentication failed", operation, false, attempt, err)
	}

	if strings.Contains(errStr, "rate limit") ||
		strings.Contains(errStr, "quota exceeded") ||
		strings.Contains(errStr, "resource exhausted") {
		return simpleai.NewLLMError(simpleai.ErrRateLimitExceeded,
			"rate limit exceeded", operation, true, attempt, err)
	}

	if strings.Contains(errStr, "model not found") ||
		strings.Contains(errStr, "invalid model") ||
		strings.Contains(errStr, "model is not available") {
		return simpleai.NewLLMError(simpleai.ErrModelNotAvailable,
			"requested model not available", operation, false, attempt, err)
	}

	// Default to retryable operation failure
	return simpleai.NewLLMError(simpleai.ErrOperationFailed,
		"operation failed", operation, true, attempt, err)
}

// pow calculates power for exponential backoff
func pow(base float64, exp int) float64 {
	result := 1.0
	for i := 0; i < exp; i++ {
		result *= base
	}
	return result
}

// extractJSON attempts to extract valid JSON from a string that may contain extra text
// Reuses the same logic from ollama/client.go
func extractJSON(response string) string {
	// Strategy 1: Try the response as-is after basic cleanup
	cleaned := strings.TrimSpace(response)
	if isValidJSON(cleaned) {
		return cleaned
	}

	// Strategy 2: Try repairing common JSON issues (e.g., unescaped newlines)
	if repaired := repairCommonJSONIssues(cleaned); isValidJSON(repaired) {
		return repaired
	}

	// Strategy 3: Check for markdown code blocks (```json ... ```)
	if strings.Contains(response, "```json") {
		if candidate := extractFromMarkdown(response, "```json"); candidate != "" {
			return candidate
		}
	}

	// Strategy 4: Check for generic code blocks (``` ... ```)
	if strings.Contains(response, "```") {
		if candidate := extractFromMarkdown(response, "```"); candidate != "" {
			return candidate
		}
	}

	// Strategy 5: Use brace counting for complex responses
	if candidate := extractWithBraceCounting(response); candidate != "" {
		return candidate
	}

	// Strategy 6: Try to find JSON-like patterns with more aggressive cleanup
	if candidate := extractWithAggressiveCleanup(response); candidate != "" {
		return candidate
	}

	return ""
}

// repairCommonJSONIssues fixes common JSON formatting issues from LLM responses
func repairCommonJSONIssues(jsonStr string) string {
	// Fix unescaped newlines in JSON strings
	repaired := fixUnescapedNewlines(jsonStr)
	return repaired
}

// fixUnescapedNewlines fixes literal newlines in JSON string values
func fixUnescapedNewlines(jsonStr string) string {
	var result strings.Builder
	inString := false
	escaped := false

	for _, char := range jsonStr {
		switch {
		case char == '\\' && !escaped:
			escaped = true
			result.WriteRune(char)
		case char == '"' && !escaped:
			inString = !inString
			result.WriteRune(char)
		case char == '\n' && inString && !escaped:
			result.WriteString("\\n")
		case char == '\r' && inString && !escaped:
			result.WriteString("\\r")
		case char == '\t' && inString && !escaped:
			result.WriteString("\\t")
		default:
			result.WriteRune(char)
		}

		if escaped && char != '\\' {
			escaped = false
		}
	}

	return result.String()
}

// isValidJSON checks if a string is valid JSON using Go's standard library
func isValidJSON(s string) bool {
	var temp interface{}
	return json.Unmarshal([]byte(s), &temp) == nil
}

// extractFromMarkdown extracts JSON from markdown code blocks
func extractFromMarkdown(response, marker string) string {
	start := strings.Index(response, marker)
	if start == -1 {
		return ""
	}

	content := response[start+len(marker):]
	end := strings.Index(content, "```")
	if end == -1 {
		return ""
	}

	candidate := strings.TrimSpace(content[:end])
	if isValidJSON(candidate) {
		return candidate
	}
	return ""
}

// extractWithBraceCounting uses brace counting to find JSON boundaries
func extractWithBraceCounting(response string) string {
	return extractJSONBoundaries(response, '{', '}')
}

// extractJSONBoundaries finds JSON object/array boundaries using character counting
func extractJSONBoundaries(response string, openChar, closeChar rune) string {
	start := -1
	count := 0
	inString := false
	escaped := false

	for i, char := range response {
		if char == '\\' && !escaped {
			escaped = true
			continue
		}

		if char == '"' && !escaped {
			inString = !inString
		}

		escaped = false

		if !inString {
			if char == openChar {
				if start == -1 {
					start = i
				}
				count++
			} else if char == closeChar {
				count--
				if count == 0 && start != -1 {
					candidate := response[start : i+1]
					if isValidJSON(candidate) {
						return candidate
					}
					start = -1
					count = 0
				}
			}
		}
	}

	return ""
}

// extractWithAggressiveCleanup tries various cleanup strategies for malformed responses
func extractWithAggressiveCleanup(response string) string {
	prefixes := []string{"```json", "```", "Here's the JSON:", "JSON:", "{", "["}
	suffixes := []string{"```", "}", "]"}

	cleaned := response

	for _, prefix := range prefixes {
		if strings.HasPrefix(cleaned, prefix) {
			cleaned = strings.TrimSpace(cleaned[len(prefix):])
			break
		}
	}

	for _, suffix := range suffixes {
		if strings.HasSuffix(cleaned, suffix) && suffix != "}" && suffix != "]" {
			cleaned = strings.TrimSpace(cleaned[:len(cleaned)-len(suffix)])
			break
		}
	}

	if isValidJSON(cleaned) {
		return cleaned
	}

	return extractJSONBoundaries(cleaned, '{', '}')
}
