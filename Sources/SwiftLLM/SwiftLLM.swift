// SwiftLLM - A protocol-based Swift package for integrating multiple LLM providers
//
// This package provides a unified interface for working with various LLM providers
// (Anthropic, OpenAI, local models, etc.) with support for:
// - Text completion
// - Structured output
// - Streaming responses
// - Token estimation
// - Provider capabilities detection

// Export public API
@_exported import struct Foundation.URL
@_exported import struct Foundation.Data

// Re-export all public types
public typealias SwiftLLM = LLMProvider
