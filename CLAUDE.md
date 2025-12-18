# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Build the package
swift build

# Run all tests
swift test

# Run a single test file
swift test --filter SwiftLLMTests

# Run benchmarks
swift test --filter SwiftLLMBenchmarks

# Run the xAI manual test executable
swift run xai-test
```

## Architecture Overview

SwiftLLM is a unified Swift 6 LLM provider library with full async/await and Sendable conformance.

### Core Protocol

`LLMProvider` (Sources/SwiftLLM/Protocols/LLMProvider.swift) defines the unified interface:
- `generateCompletion()` - Basic text completion
- `generateStructuredOutput<T: Codable>()` - JSON structured output
- `streamCompletion()` - AsyncThrowingStream for token streaming
- `generateCompletionWithTools()` / `continueWithToolResults()` - Tool calling support
- `generateGenerable<T: Generable>()` - Apple @Generable macro support (AFM only, macOS 26+)

### Provider Implementations

Each provider in `Sources/SwiftLLM/Providers/` implements `LLMProvider`:
- **AnthropicProvider** - Claude models (Opus 4.5, Sonnet 4.5, Haiku 4.5)
- **OpenAIProvider** - GPT models (GPT-5.1, GPT-5, GPT-4.x)
- **XAIProvider** - Grok models (4.1 Thinking, 4.1 Fast)
- **AppleIntelligenceProvider** - On-device/server AFM (macOS 26+)
- **LocalLLMProvider** - Ollama, LM Studio, OpenAI-compatible endpoints

### API Clients

HTTP clients in `Sources/SwiftLLM/Clients/` handle provider-specific API formats:
- **AnthropicAPIClient** - Anthropic Messages API (uses `input_schema` for tools)
- **OpenAIAPIClient** - OpenAI Chat Completions API
- **XAIAPIClient** - xAI API (OpenAI-compatible with some differences)

### Tool Calling

Tool calling types in `Sources/SwiftLLM/Models/`:
- `Tool.swift` - Tool definition with validation (snake_case names, 10-500 char descriptions)
- `ToolCall.swift` - Tool invocation from model response
- `ToolResult.swift` - Tool execution result with success/error handling
- `ToolChoice.swift` - Tool selection mode (auto/none/required/specific)
- `ConversationContext.swift` - Multi-turn conversation state management

### Provider Registry

`ProviderRegistry` (Sources/SwiftLLM/ProviderRegistry.swift) provides factory methods for creating providers and capability discovery.

## Logging

Uses Apple's unified logging (`os.log`) with subsystem `com.swiftllm`. View logs:
```bash
log stream --predicate 'subsystem == "com.swiftllm"' --level debug
```

Categories: API, Provider, Tools, General, Error

## Key Patterns

- All providers are `Sendable` for Swift 6 concurrency
- Use `async/await` throughout - no completion handlers
- Protocol extensions provide default implementations that throw `.unsupportedFeature`
- Tool names must be snake_case (validated at creation)
- `CompletionResponse.text` is non-optional; check `requiresToolExecution` for tool calls
