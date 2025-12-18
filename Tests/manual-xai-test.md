# Manual xAI Testing Guide

## Setup

1. Paste your xAI API key in `.env`:
   ```bash
   echo "XAI_API_KEY=your-actual-key-here" > .env
   ```

2. Create a simple test in the Examples directory or run directly in a Swift REPL

## Test Code

```swift
import SwiftLLM

// Load API key from environment or .env file
let apiKey = ProcessInfo.processInfo.environment["XAI_API_KEY"] ?? "your-key-from-env"

// Create xAI provider
let provider = XAIProvider.grok41FastNonReasoning(apiKey: apiKey)

// Test 1: Simple completion
print("Test 1: Simple completion")
let response = try await provider.generateCompletion(
    prompt: "Say hello in JSON format with a 'message' field",
    systemPrompt: nil,
    options: .default
)
print("Response: \(response.text)")
print("")

// Test 2: Structured output (tests markdown stripping)
print("Test 2: Structured output")
struct Greeting: Codable {
    let message: String
    let language: String
}

let structured = try await provider.generateStructuredOutput(
    prompt: "Generate a greeting in English",
    systemPrompt: nil,
    schema: Greeting.self,
    options: .default
)
print("Structured: \(structured.message) (\(structured.language))")
print("")

print("✅ All tests passed!")
```

## Issues to Watch For

1. **Markdown code blocks**: xAI wraps JSON responses in:
   ```json
   {
     "message": "Hello!"
   }
   ```
   The `stripMarkdownCodeBlocks()` function should handle this.

2. **Model availability**: Ensure you're using an available model:
   - `grok-4-1-fast-non-reasoning` (recommended)
   - `grok-4-1-fast-reasoning`
   - `grok-4-1-thinking`
   - `grok-4-1`

3. **Error messages**: Check if errors provide useful context about the response format.
