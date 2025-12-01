# SwiftLLM: Adding Native Apple `@Generable` Support

## Problem

`AppleIntelligenceProvider.generateStructuredOutput<T: Codable>` uses a JSON prompt hack that fails because AFM returns markdown-wrapped JSON. Apple's proper solution is the `@Generable` macro which constrains the model to output valid schema directly.

## Solution: Add Parallel Generable API

### 1. New protocol method in `LLMProvider`:

```swift
@available(macOS 26.0, *)
func generateGenerable<T: Generable>(
    prompt: String,
    systemPrompt: String?,
    responseType: T.Type,
    options: GenerationOptions
) async throws -> T
```

### 2. AFM implementation uses native guided generation:

```swift
let response = try await session.respond(
    to: prompt,
    generating: T.self,  // Generable constraint
    options: genOptions
)
return response.content  // Already typed as T, no JSON parsing
```

### 3. App-side: Define response types with macros:

```swift
@Generable
struct CodeSummaryResponse {
    @Guide(description: "One-line summary")
    var brief: String

    @Guide(description: "Key points about the code")
    var keyPoints: [String]
}
```

### 4. Fallback

Other providers (Anthropic, OpenAI) continue using the existing `generateStructuredOutput<T: Codable>` with JSON schema. The app chooses which method based on provider capabilities.

## Key Benefit

No JSON parsing, no markdown stripping - AFM internally constrains token generation to match the schema, returning native Swift objects directly.
