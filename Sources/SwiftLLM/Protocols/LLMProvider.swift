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
}
