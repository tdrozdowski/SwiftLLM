import Foundation

/// Describes what features an LLM provider supports
public struct LLMCapabilities: Sendable, Codable, Equatable {
    /// Can generate structured JSON outputs
    public let supportsStructuredOutput: Bool

    /// Can stream responses token-by-token
    public let supportsStreaming: Bool

    /// Can run locally (no network required)
    public let supportsLocalExecution: Bool

    /// Can handle image inputs (vision models)
    public let supportsVision: Bool

    /// Can use function/tool calling
    public let supportsToolCalling: Bool

    /// Maximum context window size (tokens)
    public let maxContextTokens: Int

    /// Maximum output tokens per request
    public let maxOutputTokens: Int

    /// Supports system prompts
    public let supportsSystemPrompts: Bool

    /// Cost information (if applicable)
    public let pricing: LLMPricing?

    public init(
        supportsStructuredOutput: Bool = false,
        supportsStreaming: Bool = false,
        supportsLocalExecution: Bool = false,
        supportsVision: Bool = false,
        supportsToolCalling: Bool = false,
        maxContextTokens: Int,
        maxOutputTokens: Int,
        supportsSystemPrompts: Bool = true,
        pricing: LLMPricing? = nil
    ) {
        self.supportsStructuredOutput = supportsStructuredOutput
        self.supportsStreaming = supportsStreaming
        self.supportsLocalExecution = supportsLocalExecution
        self.supportsVision = supportsVision
        self.supportsToolCalling = supportsToolCalling
        self.maxContextTokens = maxContextTokens
        self.maxOutputTokens = maxOutputTokens
        self.supportsSystemPrompts = supportsSystemPrompts
        self.pricing = pricing
    }
}

/// Pricing information for an LLM provider
public struct LLMPricing: Sendable, Codable, Equatable {
    /// Cost per 1M input tokens (USD)
    public let inputCostPer1M: Double

    /// Cost per 1M output tokens (USD)
    public let outputCostPer1M: Double

    public init(inputCostPer1M: Double, outputCostPer1M: Double) {
        self.inputCostPer1M = inputCostPer1M
        self.outputCostPer1M = outputCostPer1M
    }
}
