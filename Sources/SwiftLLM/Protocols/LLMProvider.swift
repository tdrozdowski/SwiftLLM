import Foundation

/// Core protocol that all LLM providers must implement
public protocol LLMProvider: Sendable {
    /// Unique identifier for this provider (e.g., "anthropic", "openai")
    var id: String { get }

    /// Display name for UI (e.g., "Anthropic Claude", "OpenAI GPT-4")
    var displayName: String { get }

    /// What capabilities this provider supports
    var capabilities: LLMCapabilities { get }

    /// Generate a text completion
    /// - Parameters:
    ///   - prompt: The user prompt/message
    ///   - systemPrompt: Optional system instructions
    ///   - options: Generation configuration (temperature, tokens, etc.)
    /// - Returns: The completion response
    func generateCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) async throws -> CompletionResponse

    /// Generate structured output conforming to a schema
    /// - Parameters:
    ///   - prompt: The user prompt/message
    ///   - systemPrompt: Optional system instructions
    ///   - schema: The Codable type to decode into
    ///   - options: Generation configuration
    /// - Returns: Decoded structured output
    func generateStructuredOutput<T: Codable>(
        prompt: String,
        systemPrompt: String?,
        schema: T.Type,
        options: GenerationOptions
    ) async throws -> T

    /// Stream completion tokens as they arrive
    /// - Parameters:
    ///   - prompt: The user prompt/message
    ///   - systemPrompt: Optional system instructions
    ///   - options: Generation configuration
    /// - Returns: Async stream of text chunks
    func streamCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) -> AsyncThrowingStream<String, Error>

    /// Estimate token count for a prompt
    /// - Parameter text: The text to count tokens for
    /// - Returns: Estimated token count
    func estimateTokens(_ text: String) async throws -> Int

    /// Generate completion with tool support
    /// - Parameters:
    ///   - context: Conversation context with tools and messages
    ///   - toolChoice: How the model should choose tools
    ///   - options: Generation configuration
    /// - Returns: Completion response with potential tool calls
    /// - Note: Default implementation throws `.unsupportedFeature` if not overridden
    func generateCompletionWithTools(
        context: ConversationContext,
        toolChoice: ToolChoice,
        options: GenerationOptions
    ) async throws -> CompletionResponse

    /// Continue conversation after tool execution
    /// - Parameters:
    ///   - context: Conversation context with tool results
    ///   - options: Generation configuration
    /// - Returns: Completion response
    /// - Note: Default implementation throws `.unsupportedFeature` if not overridden
    func continueWithToolResults(
        context: ConversationContext,
        options: GenerationOptions
    ) async throws -> CompletionResponse
}

// MARK: - Default Implementations

public extension LLMProvider {
    func generateCompletionWithTools(
        context: ConversationContext,
        toolChoice: ToolChoice = .auto,
        options: GenerationOptions = .default
    ) async throws -> CompletionResponse {
        guard capabilities.supportsToolCalling else {
            throw LLMError.unsupportedFeature("Tool calling is not supported by \(displayName)")
        }
        // Default implementation for providers that haven't implemented tool support yet
        throw LLMError.unsupportedFeature("Tool calling not yet implemented for \(displayName)")
    }

    func continueWithToolResults(
        context: ConversationContext,
        options: GenerationOptions = .default
    ) async throws -> CompletionResponse {
        guard capabilities.supportsToolCalling else {
            throw LLMError.unsupportedFeature("Tool calling is not supported by \(displayName)")
        }
        throw LLMError.unsupportedFeature("Tool calling not yet implemented for \(displayName)")
    }
}

// MARK: - Apple Foundation Models Extensions

#if canImport(FoundationModels)
import FoundationModels

@available(macOS 26.0, iOS 26.0, *)
public extension LLMProvider {
    /// Generate a response with native Apple @Generable type
    ///
    /// This method is only available for providers that support Apple's @Generable macro
    /// (currently only AppleIntelligenceProvider). For other providers, this will throw
    /// `.unsupportedFeature`.
    ///
    /// - Parameters:
    ///   - prompt: The user's prompt
    ///   - systemPrompt: Optional system instructions
    ///   - responseType: The @Generable type to generate
    ///   - options: Generation options (temperature, max tokens, etc.)
    /// - Returns: The generated response as the specified @Generable type
    /// - Throws: LLMError.unsupportedFeature for non-AFM providers, or generation errors
    ///
    /// - Note: Default implementation throws `.unsupportedFeature`. Only AppleIntelligenceProvider
    ///         implements this method.
    func generateGenerable<T: Generable>(
        prompt: String,
        systemPrompt: String?,
        responseType: T.Type,
        options: GenerationOptions = .default
    ) async throws -> T {
        throw LLMError.unsupportedFeature(
            "Native @Generable support is only available with Apple Intelligence (AppleIntelligenceProvider). " +
            "Use generateStructuredOutput<T: Codable> instead for \(displayName)."
        )
    }
}
#endif
