# SwiftLLM

A modern, protocol-based Swift package for integrating multiple LLM providers with a unified, type-safe API. Built for Apple platforms with full Swift 6 concurrency support.

[![Swift Version](https://img.shields.io/badge/Swift-6.0+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%2013+%20|%20iOS%2016+-blue.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-Private-red.svg)](LICENSE)

## Overview

SwiftLLM provides a unified interface for working with the latest LLM providers, supporting all major models released in 2025 including Claude 4.5, GPT-5.1, Grok 4.1, and Apple Foundation Models.

## Features

- 🔌 **Multi-Provider Support**: Anthropic Claude, OpenAI GPT, xAI Grok, Apple Intelligence
- 🎯 **Type-Safe**: Leverage Swift's type system for structured outputs
- ⚡ **Modern Concurrency**: Built with async/await and actors throughout
- 🌊 **Streaming Support**: Real-time token-by-token responses
- 🔍 **Capability Detection**: Query what each provider supports
- 📊 **Usage Tracking**: Monitor token usage and costs
- 🔐 **Privacy-First**: On-device Apple Intelligence support
- 🎨 **Latest Models**: Support for November 2025 frontier models
- 🛡️ **Swift 6 Ready**: Full Sendable conformance and strict concurrency

## Supported Providers

### Anthropic Claude
- ✅ Claude Opus 4.5 (flagship model, best for coding/agents)
- ✅ Claude Sonnet 4.5 (best coding model)
- ✅ Claude Haiku 4.5 (fast, low-cost)
- ✅ Claude 4.x series (Opus 4.1, Sonnet 4, Opus 4)
- ✅ Claude 3.x series (legacy support)

### OpenAI GPT
- ✅ GPT-5.1 Instant (adaptive reasoning)
- ✅ GPT-5.1 Thinking (advanced reasoning)
- ✅ GPT-5.1-Codex-Max (frontier agentic coding)
- ✅ GPT-5.1-Codex and Codex-Mini
- ✅ GPT-5 (August 2025 frontier model)
- ✅ GPT-4.x series (legacy support)

### xAI Grok
- ✅ Grok 4.1 Thinking (#1 on LMArena)
- ✅ Grok 4.1 Fast (2M context window)
- ✅ Grok 4.1 standard
- ✅ Grok 2.x series (legacy support)

### Apple Foundation Models
- ✅ On-Device AFM (3B params, private, offline)
- ✅ Server-Based AFM (larger MoE, 32K context)
- ✅ Native FoundationModels framework integration (macOS 26+)

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/tdrozdowski/SwiftLLM.git", from: "1.0.0")
]
```

Or add via Xcode: **File → Add Package Dependencies** and enter:
```
https://github.com/tdrozdowski/SwiftLLM.git
```

## Quick Start

### Anthropic Claude

```swift
import SwiftLLM

// Use the latest Claude Opus 4.5
let provider = AnthropicProvider.opus45(apiKey: "your-api-key")

let response = try await provider.generateCompletion(
    prompt: "Explain quantum entanglement in simple terms",
    systemPrompt: "You are a physics teacher",
    options: GenerationOptions(temperature: 0.7)
)

print(response.text)
```

### OpenAI GPT

```swift
// Use GPT-5.1 Instant with adaptive reasoning
let provider = OpenAIProvider.gpt51Instant(apiKey: "your-api-key")

let response = try await provider.generateCompletion(
    prompt: "Write a Python function to find prime numbers",
    systemPrompt: "You are an expert programmer",
    options: GenerationOptions(temperature: 0.3, maxTokens: 1000)
)

print(response.text)
```

### xAI Grok

```swift
// Use Grok 4.1 Thinking (top-ranked model)
let provider = XAIProvider.grok41Thinking(apiKey: "your-api-key")

let response = try await provider.generateCompletion(
    prompt: "Analyze the implications of quantum computing on cryptography",
    systemPrompt: nil,
    options: GenerationOptions(temperature: 0.5)
)

print(response.text)
```

### Apple Foundation Models

```swift
// On-device (private, offline, free)
let onDevice = AppleIntelligenceProvider.onDevice(
    instructions: "You are a helpful writing assistant"
)

// Server-based (more powerful, requires network)
let server = AppleIntelligenceProvider.server(
    instructions: "You are an expert software architect"
)

let response = try await onDevice.generateCompletion(
    prompt: "Draft a professional email about...",
    systemPrompt: nil,
    options: GenerationOptions(temperature: 0.7)
)
```

## Advanced Usage

### Streaming Responses

```swift
let provider = AnthropicProvider.sonnet45(apiKey: "your-api-key")

let stream = provider.streamCompletion(
    prompt: "Write a short story about a time traveler",
    systemPrompt: "You are a creative writer",
    options: GenerationOptions(temperature: 0.9)
)

for try await chunk in stream {
    print(chunk, terminator: "")
}
```

### Structured Output

```swift
struct Recipe: Codable {
    let name: String
    let ingredients: [String]
    let steps: [String]
    let prepTime: Int
    let servings: Int
}

let provider = OpenAIProvider.gpt51Instant(apiKey: "your-api-key")

let recipe = try await provider.generateStructuredOutput(
    prompt: "Create a recipe for chocolate chip cookies",
    systemPrompt: "You are a professional chef",
    schema: Recipe.self,
    options: GenerationOptions(temperature: 0.7)
)

print("Recipe: \(recipe.name)")
print("Ingredients: \(recipe.ingredients.joined(separator: ", "))")
```

### Checking Provider Capabilities

```swift
let provider = AppleIntelligenceProvider.onDevice()

print("Supports streaming: \(provider.capabilities.supportsStreaming)")
print("Supports vision: \(provider.capabilities.supportsVision)")
print("Max context: \(provider.capabilities.maxContextTokens) tokens")
print("Runs locally: \(provider.capabilities.supportsLocalExecution)")

if let pricing = provider.capabilities.pricing {
    print("Input cost: $\(pricing.inputCostPer1M) per 1M tokens")
    print("Output cost: $\(pricing.outputCostPer1M) per 1M tokens")
} else {
    print("Free (on-device execution)")
}
```

### Token Usage Tracking

```swift
let response = try await provider.generateCompletion(
    prompt: "Your prompt here",
    systemPrompt: nil,
    options: GenerationOptions()
)

print("Tokens used:")
print("  Input: \(response.usage.inputTokens)")
print("  Output: \(response.usage.outputTokens)")
print("  Total: \(response.usage.totalTokens)")

// Calculate cost if pricing is available
if let pricing = provider.capabilities.pricing {
    let inputCost = Double(response.usage.inputTokens) * pricing.inputCostPer1M / 1_000_000
    let outputCost = Double(response.usage.outputTokens) * pricing.outputCostPer1M / 1_000_000
    let totalCost = inputCost + outputCost
    print("  Cost: $\(String(format: "%.4f", totalCost))")
}
```

### Error Handling

```swift
do {
    let response = try await provider.generateCompletion(
        prompt: "Your prompt",
        systemPrompt: nil,
        options: GenerationOptions()
    )
    print(response.text)
} catch LLMError.authenticationFailed(let message) {
    print("Authentication failed: \(message)")
} catch LLMError.rateLimitExceeded(let retryAfter) {
    if let retryTime = retryAfter {
        print("Rate limited. Retry after \(retryTime) seconds")
    } else {
        print("Rate limited. Try again later")
    }
} catch LLMError.invalidRequest(let message) {
    print("Invalid request: \(message)")
} catch LLMError.providerError(let message, let code) {
    print("Provider error [\(code ?? "unknown")]: \(message)")
} catch {
    print("Unexpected error: \(error)")
}
```

## Provider Selection Guide

### Choose Anthropic Claude When:
- ✅ You need best-in-class coding assistance (Opus 4.5)
- ✅ Building agents or using computer use
- ✅ Need strong reasoning and math (Sonnet 4.5)
- ✅ Want high-quality output with good context understanding
- **Cost**: $1-25/1M tokens depending on model

### Choose OpenAI GPT When:
- ✅ Need adaptive reasoning (GPT-5.1 Instant)
- ✅ Building complex agentic coding systems (Codex-Max)
- ✅ Want vision capabilities
- ✅ Need structured outputs with JSON mode
- **Cost**: $5-30/1M tokens depending on model

### Choose xAI Grok When:
- ✅ Need the highest-ranked model (Grok 4.1 Thinking)
- ✅ Require massive context (2M tokens for Fast models)
- ✅ Want lowest hallucination rate (3x better than previous)
- ✅ Budget-conscious projects (Fast models at $0.20-0.50/1M)
- **Cost**: $0.20-10/1M tokens depending on model

### Choose Apple AFM When:
- ✅ Privacy is paramount (on-device processing)
- ✅ Need offline capabilities
- ✅ Want zero cost (on-device)
- ✅ Building for Apple platforms exclusively
- ✅ Need real-time, low-latency responses
- **Cost**: Free (on-device), TBD (server)

See [Documentation/AppleFoundationModels.md](Documentation/AppleFoundationModels.md) for detailed AFM guidance.

## Architecture

```
SwiftLLM/
├── Protocols/
│   └── LLMProvider.swift          # Core protocol defining LLM interface
├── Models/
│   ├── LLMCapabilities.swift      # Feature detection & pricing
│   ├── GenerationOptions.swift    # Configuration for generation
│   ├── CompletionResponse.swift   # Response with usage tracking
│   └── LLMError.swift             # Comprehensive error handling
├── Providers/
│   ├── AnthropicProvider.swift    # Claude 4.5, 4.x, 3.x support
│   ├── OpenAIProvider.swift       # GPT-5.1, 5, 4.x support
│   ├── XAIProvider.swift          # Grok 4.1, 2.x support
│   └── AppleIntelligenceProvider.swift  # AFM on-device & server
├── Clients/
│   ├── AnthropicAPIClient.swift   # Anthropic Messages API
│   ├── OpenAIAPIClient.swift      # OpenAI Chat Completions API
│   └── XAIAPIClient.swift         # xAI (OpenAI-compatible) API
└── Documentation/
    └── AppleFoundationModels.md   # AFM usage guide
```

## Requirements

- **Swift**: 6.0+
- **Platforms**: macOS 13+, iOS 16+
- **Apple Intelligence**: macOS 26+, iOS 26+ (for AFM)

## Documentation

- [Apple Foundation Models Guide](Documentation/AppleFoundationModels.md) - Detailed AFM usage, limitations, and best practices
- API Documentation (coming soon via DocC)

## Examples

See the [Examples](Examples/) directory for:
- Basic usage examples
- Streaming responses
- Structured output
- Error handling
- Multi-provider comparison
- Real-world use cases

## Roadmap

### Completed ✅
- [x] Core protocol and models
- [x] Anthropic Claude provider (4.5, 4.x, 3.x)
- [x] OpenAI GPT provider (5.1, 5, 4.x)
- [x] xAI Grok provider (4.1, 2.x)
- [x] Apple Intelligence integration (AFM)
- [x] Streaming support
- [x] Structured output
- [x] Vision model support (GPT, Claude, AFM)
- [x] Tool calling capabilities

### Planned 🚧
- [ ] Ollama (local) provider
- [ ] Prompt caching support
- [ ] Multi-modal inputs (images, audio)
- [ ] Function/tool calling helpers
- [ ] Usage analytics and cost tracking
- [ ] Provider factory pattern
- [ ] SwiftUI integration helpers
- [ ] Performance benchmarks

## Testing

Run tests with:
```bash
swift test
```

## Contributing

This is currently a private package for personal use. Contact the author for collaboration inquiries.

## License

Private - For personal use only

## Author

**Terry Drozdowski**
- GitHub: [@tdrozdowski](https://github.com/tdrozdowski)

## Acknowledgments

Built with the latest models from:
- [Anthropic Claude](https://www.anthropic.com) - Claude Opus 4.5, Sonnet 4.5, Haiku 4.5
- [OpenAI](https://openai.com) - GPT-5.1 series
- [xAI](https://x.ai) - Grok 4.1 series
- [Apple](https://developer.apple.com/documentation/foundationmodels) - Foundation Models framework

---

**Note**: This package requires API keys from the respective providers (except Apple AFM). Keep your API keys secure and never commit them to version control.
