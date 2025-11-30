# SwiftLLM Examples

This directory contains practical examples demonstrating how to use SwiftLLM with different providers.

## Running Examples

### Prerequisites

1. **API Keys**: You'll need API keys from the providers you want to test:
   - Anthropic: https://console.anthropic.com/
   - OpenAI: https://platform.openai.com/api-keys
   - xAI: https://x.ai/api

2. **Local LLMs** (no API key needed):
   - Ollama: https://ollama.ai (run `ollama serve` and `ollama pull llama3.2`)
   - LM Studio: https://lmstudio.ai (download a model and start the server)

3. **macOS 26+ / iOS 26+** (for Apple Foundation Models examples only)

### Basic Usage

The `BasicUsage.swift` file contains examples for all providers:

```swift
// Uncomment the examples you want to run in BasicUsage.swift
try await anthropicBasicExample()
try await openAIBasicExample()
try await xAIBasicExample()

// Local LLMs (no API key needed - just run Ollama/LM Studio)
try await ollamaBasicExample()
try await lmStudioExample()

// On macOS 26+ / iOS 26+
if #available(macOS 26.0, iOS 26.0, *) {
    try await appleOnDeviceExample()
}
```

## Examples Included

### Anthropic Claude

1. **Basic Completion** (`anthropicBasicExample`)
   - Simple text generation with Claude Opus 4.5
   - Shows token usage tracking

2. **Streaming** (`anthropicStreamingExample`)
   - Real-time streaming with Claude Sonnet 4.5
   - Demonstrates token-by-token output

### OpenAI GPT

3. **Basic Completion** (`openAIBasicExample`)
   - Code generation with GPT-5.1 Instant
   - Shows model and finish reason

4. **Structured Output** (`openAIStructuredOutputExample`)
   - Type-safe JSON responses
   - Custom Codable schemas

### xAI Grok

5. **Grok Thinking** (`xAIBasicExample`)
   - Complex reasoning with Grok 4.1 Thinking
   - Cost calculation example

6. **Grok Fast** (`xAIFastExample`)
   - High-speed responses with 2M context window
   - Capability inspection

### Apple Foundation Models

7. **On-Device** (`appleOnDeviceExample`)
   - Private, offline processing
   - Zero-cost on-device execution

8. **Server-Based** (`appleServerExample`)
   - More powerful cloud model
   - Larger context window (32K)

### Local LLMs (Ollama, LM Studio, etc.)

9. **Ollama Basic** (`ollamaBasicExample`)
   - Simple local model usage
   - No API key required

10. **Ollama Custom Model** (`ollamaCustomModelExample`)
    - Full control with `LocalModelConfig`
    - Custom context window, capabilities

11. **LM Studio** (`lmStudioExample`)
    - Streaming with LM Studio
    - Default port 1234

12. **OpenAI-Compatible Server** (`openAICompatibleServerExample`)
    - Works with vLLM, text-generation-inference, FastChat
    - Custom endpoints with optional API key

13. **Local Structured Output** (`localLLMStructuredOutputExample`)
    - Type-safe JSON responses from local models
    - Code review example

### Utilities

14. **Error Handling** (`errorHandlingExample`)
    - Comprehensive error handling patterns
    - All LLMError cases covered

15. **Provider Comparison** (`providerComparisonExample`)
    - Side-by-side provider testing
    - Performance and cost comparison

## Security Note

**Never commit API keys to version control!**

Best practices:
- Use environment variables
- Store in macOS Keychain
- Use a `.env` file (add to `.gitignore`)

Example with environment variables:

```swift
let apiKey = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] ?? ""
let provider = AnthropicProvider.opus45(apiKey: apiKey)
```

## Next Steps

After trying these examples, explore:
- Custom provider configurations
- Multi-turn conversations
- Image/vision inputs (for supported providers)
- Tool calling (advanced feature)
- Running your own fine-tuned models with Ollama or LM Studio
