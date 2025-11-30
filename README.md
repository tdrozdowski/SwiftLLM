# SwiftLLM

A protocol-based Swift package for integrating multiple LLM providers with a clean, type-safe API.

## Overview

SwiftLLM provides a unified interface for working with various LLM providers (Anthropic Claude, OpenAI GPT, local models via Ollama, etc.) across Apple platforms and Linux.

## Features

- 🔌 **Provider Abstraction**: Protocol-based design works with any LLM provider
- 🎯 **Type-Safe**: Leverage Swift's type system for structured outputs
- ⚡ **Modern Concurrency**: Built with async/await throughout
- 🌊 **Streaming Support**: Token-by-token streaming responses
- 🔍 **Capability Detection**: Query what each provider supports
- 📊 **Usage Tracking**: Monitor token usage and costs
- 🌐 **Cross-Platform**: Works on macOS, iOS, and Linux

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/tdrozdowski/SwiftLLM.git", from: "1.0.0")
]
```

Or add via Xcode: **File → Add Package Dependencies**

## Usage

### Basic Text Completion

```swift
import SwiftLLM

let provider: any LLMProvider = // ... your provider

let response = try await provider.generateCompletion(
    prompt: "Explain quantum computing in simple terms",
    systemPrompt: "You are a helpful teacher",
    options: .default
)

print(response.text)
```

### Structured Output

```swift
struct Recipe: Codable {
    let name: String
    let ingredients: [String]
    let steps: [String]
}

let recipe = try await provider.generateStructuredOutput(
    prompt: "Create a recipe for chocolate chip cookies",
    systemPrompt: nil,
    schema: Recipe.self,
    options: .default
)

print(recipe.name)
```

### Streaming Responses

```swift
let stream = provider.streamCompletion(
    prompt: "Write a story about a robot",
    systemPrompt: nil,
    options: .default
)

for try await chunk in stream {
    print(chunk, terminator: "")
}
```

## Architecture

```
SwiftLLM/
├── Protocols/
│   └── LLMProvider.swift      # Core protocol
├── Models/
│   ├── LLMCapabilities.swift  # Feature detection
│   ├── GenerationOptions.swift
│   ├── CompletionResponse.swift
│   └── LLMError.swift
├── Providers/                 # Concrete implementations (coming soon)
│   ├── AnthropicProvider.swift
│   ├── OpenAIProvider.swift
│   └── OllamaProvider.swift
└── Clients/                   # API clients (coming soon)
    ├── AnthropicAPIClient.swift
    └── OpenAIAPIClient.swift
```

## Roadmap

- [ ] Anthropic Claude provider
- [ ] OpenAI GPT provider
- [ ] Ollama (local) provider
- [ ] Apple Intelligence integration
- [ ] Tool/function calling support
- [ ] Vision model support
- [ ] Prompt caching
- [ ] Response streaming with server-sent events

## Requirements

- Swift 6.0+
- macOS 13+, iOS 16+, or Linux

## License

Private - For personal use only

## Author

Terry Drozdowski
