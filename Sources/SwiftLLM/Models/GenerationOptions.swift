import Foundation

/// Configuration options for LLM text generation
public struct GenerationOptions: Sendable, Equatable {
    /// Model to use (provider-specific, e.g., "claude-3-opus", "gpt-4")
    public let model: String?

    /// Sampling temperature (0.0 to 1.0+)
    /// Lower = more deterministic, Higher = more creative
    public let temperature: Double?

    /// Maximum tokens to generate
    public let maxTokens: Int?

    /// Top-p (nucleus) sampling threshold
    public let topP: Double?

    /// Frequency penalty (-2.0 to 2.0)
    public let frequencyPenalty: Double?

    /// Presence penalty (-2.0 to 2.0)
    public let presencePenalty: Double?

    /// Stop sequences to halt generation
    public let stopSequences: [String]?

    /// Additional provider-specific parameters
    public let customParameters: [String: String]?

    public init(
        model: String? = nil,
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        stopSequences: [String]? = nil,
        customParameters: [String: String]? = nil
    ) {
        self.model = model
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.stopSequences = stopSequences
        self.customParameters = customParameters
    }

    /// Default options with sensible defaults
    public static let `default` = GenerationOptions(
        temperature: 0.7,
        maxTokens: 2048
    )
}
