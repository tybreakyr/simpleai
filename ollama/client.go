package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"simpleai"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

type Client struct {
	ollamaClient *api.Client
	model        simpleai.Model
}

func NewClient(model simpleai.Model) *Client {
	return &Client{
		ollamaClient: api.NewClient(&url.URL{Scheme: "http", Host: "localhost:11434"}, http.DefaultClient),
		model:        model,
	}
}

func PrependSystemPrompt(messages []simpleai.Message, systemPrompt simpleai.SystemPrompt) []simpleai.Message {
	return append([]simpleai.Message{{Role: "system", Content: systemPrompt.Content}}, messages...)
}

func ConvertMessages(messages []simpleai.Message) []api.Message {
	converted := make([]api.Message, len(messages))
	for i, msg := range messages {
		converted[i] = api.Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}
	return converted
}

func (c *Client) Chat(request simpleai.ChatRequest) (simpleai.ChatResponse, error) {
	return c.ChatWithRetry(request, simpleai.DefaultRetryConfig())
}

// ChatWithRetry executes a chat request with retry logic
func (c *Client) ChatWithRetry(request simpleai.ChatRequest, retryConfig *simpleai.RetryConfig) (simpleai.ChatResponse, error) {
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

		handler := func(resp api.ChatResponse) error {
			finalResponse += resp.Message.Content
			return nil
		}

		stream := false
		thinkValue := &api.ThinkValue{
			Value: false,
		}

		// Use configurable timeout (default 60 seconds, but could be made configurable)
		timeout := 60 * time.Second
		ctx, cancel := context.WithTimeout(context.Background(), timeout)

		err := c.ollamaClient.Chat(ctx, &api.ChatRequest{
			Model:    c.model.Name,
			Messages: ConvertMessages(PrependSystemPrompt(request.Messages, request.SystemPrompt)),
			Stream:   &stream,
			Think:    thinkValue,
		}, handler)

		cancel() // Always cancel context

		if err != nil {
			lastErr = c.classifyError(err, attempt, "chat")
			if attempt < retryConfig.MaxRetries && simpleai.IsRetryable(lastErr) {
				continue // Retry
			}
			break // No more retries or non-retryable error
		}

		// Handle structured data parsing after complete response
		if targetType != nil {
			// Try to extract JSON from the response
			jsonStr := extractJSON(finalResponse)
			if jsonStr == "" {
				// Enhanced error reporting for JSON extraction failures
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
				// Enhanced error reporting for JSON unmarshaling failures
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

// classifyError classifies errors and determines if they are retryable
func (c *Client) classifyError(err error, attempt int, operation string) *simpleai.LLMError {
	errStr := err.Error()

	// Check for specific error types
	if strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "no such host") ||
		strings.Contains(errStr, "network is unreachable") {
		return simpleai.NewLLMError(simpleai.ErrConnectionFailed,
			"connection to Ollama service failed", operation, true, attempt, err)
	}

	if strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "context deadline exceeded") {
		return simpleai.NewLLMError(simpleai.ErrTimeout,
			"request timed out", operation, true, attempt, err)
	}

	if strings.Contains(errStr, "model not found") ||
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
	// This is the most common issue where LLMs put literal newlines in JSON strings
	repaired := fixUnescapedNewlines(jsonStr)

	// Fix unescaped quotes (though this is trickier and less common)
	// We'll be conservative here to avoid breaking valid JSON

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
			// Replace unescaped newline in string with escaped version
			result.WriteString("\\n")
		case char == '\r' && inString && !escaped:
			// Replace unescaped carriage return in string with escaped version
			result.WriteString("\\r")
		case char == '\t' && inString && !escaped:
			// Replace unescaped tab in string with escaped version
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

	// Move past the marker
	content := response[start+len(marker):]

	// Find the closing ```
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
		// Handle string escaping to avoid counting braces inside strings
		if char == '\\' && !escaped {
			escaped = true
			continue
		}

		if char == '"' && !escaped {
			inString = !inString
		}

		escaped = false

		// Only count braces outside of strings
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
					// Reset for next potential JSON object
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
	// Remove common prefixes that LLMs sometimes add
	prefixes := []string{"```json", "```", "Here's the JSON:", "JSON:", "{", "["}
	suffixes := []string{"```", "}", "]"}

	cleaned := response

	// Remove prefixes
	for _, prefix := range prefixes {
		if strings.HasPrefix(cleaned, prefix) {
			cleaned = strings.TrimSpace(cleaned[len(prefix):])
			break
		}
	}

	// Remove suffixes
	for _, suffix := range suffixes {
		if strings.HasSuffix(cleaned, suffix) && suffix != "}" && suffix != "]" {
			cleaned = strings.TrimSpace(cleaned[:len(cleaned)-len(suffix)])
			break
		}
	}

	// Try to find JSON boundaries again after cleanup
	if isValidJSON(cleaned) {
		return cleaned
	}

	// Try brace counting on cleaned content
	return extractJSONBoundaries(cleaned, '{', '}')
}

