import Foundation

/// Provider implementation for Anthropic Claude models
public struct AnthropicProvider: LLMProvider {
    public let id: String = "anthropic"
    public let displayName: String
    private let client: AnthropicAPIClient
    private let defaultModel: String

    public var capabilities: LLMCapabilities {
        LLMCapabilities(
            supportsStructuredOutput: false, // Claude doesn't have native structured output yet
            supportsStreaming: true,
            supportsLocalExecution: false,
            supportsVision: modelSupportsVision,
            supportsToolCalling: true,
            maxContextTokens: modelContextWindow,
            maxOutputTokens: modelMaxOutput,
            supportsSystemPrompts: true,
            pricing: modelPricing
        )
    }

    private var modelSupportsVision: Bool {
        // Claude 3.x and Claude 4.x models support vision
        defaultModel.contains("claude-3") ||
        defaultModel.contains("claude-4") ||
        defaultModel.contains("-4-")  // Matches 4-5, 4-1, etc.
    }

    private var modelContextWindow: Int {
        // Claude 4.x and 3.x models have 200K context window
        if defaultModel.contains("claude-3") ||
           defaultModel.contains("claude-4") ||
           defaultModel.contains("-4-") {  // Matches 4-5, 4-1, etc.
            return 200_000
        } else {
            return 100_000
        }
    }

    private var modelMaxOutput: Int {
        // Claude 4.x and 3.5 models have 8K max output
        if defaultModel.contains("claude-3-5") ||
           defaultModel.contains("claude-4") ||
           defaultModel.contains("-4-") {  // Matches 4-5, 4-1, etc.
            return 8192
        } else {
            return 4096
        }
    }

    private var modelPricing: LLMPricing? {
        // Claude 4.5 pricing (as of November 2025)
        if defaultModel.contains("opus-4-5") {
            return LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 25.0)
        } else if defaultModel.contains("sonnet-4-5") {
            return LLMPricing(inputCostPer1M: 3.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("haiku-4-5") {
            return LLMPricing(inputCostPer1M: 1.0, outputCostPer1M: 5.0)
        } else if defaultModel.contains("opus-4") {
            return LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 25.0)
        } else if defaultModel.contains("sonnet-4") {
            return LLMPricing(inputCostPer1M: 3.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("haiku-4") {
            return LLMPricing(inputCostPer1M: 1.0, outputCostPer1M: 5.0)
        } else if defaultModel.contains("claude-3-5-sonnet") {
            return LLMPricing(inputCostPer1M: 3.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("claude-3-opus") {
            return LLMPricing(inputCostPer1M: 15.0, outputCostPer1M: 75.0)
        } else if defaultModel.contains("claude-3-haiku") {
            return LLMPricing(inputCostPer1M: 0.25, outputCostPer1M: 1.25)
        }
        return nil
    }

    public init(apiKey: String, model: String = "claude-opus-4-5-20251124") {
        self.client = AnthropicAPIClient(apiKey: apiKey)
        self.defaultModel = model
        self.displayName = "Anthropic Claude"
    }

    public func generateCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) async throws -> CompletionResponse {
        let request = AnthropicAPIClient.MessagesRequest(
            model: options.model ?? defaultModel,
            messages: [
                .init(role: "user", content: prompt)
            ],
            system: systemPrompt,
            max_tokens: options.maxTokens ?? 4096,
            temperature: options.temperature,
            top_p: options.topP,
            stream: false
        )

        let response = try await client.createMessage(request: request)

        guard let textContent = response.content.first(where: { $0.type == "text" })?.text else {
            throw LLMError.decodingError("No text content in response")
        }

        return CompletionResponse(
            text: textContent,
            model: response.model,
            usage: TokenUsage(
                inputTokens: response.usage.input_tokens,
                outputTokens: response.usage.output_tokens
            ),
            finishReason: response.stop_reason,
            metadata: ["id": response.id, "type": response.type]
        )
    }

    public func generateStructuredOutput<T: Codable>(
        prompt: String,
        systemPrompt: String?,
        schema: T.Type,
        options: GenerationOptions
    ) async throws -> T {
        // For now, use prompt engineering to get JSON output
        let jsonPrompt = """
        \(prompt)

        Respond with valid JSON matching this structure. Only output the JSON, no other text.
        """

        let response = try await generateCompletion(
            prompt: jsonPrompt,
            systemPrompt: systemPrompt,
            options: options
        )

        // Try to extract JSON from response
        guard let jsonData = response.text.data(using: .utf8) else {
            throw LLMError.decodingError("Could not convert response to data")
        }

        do {
            return try JSONDecoder().decode(T.self, from: jsonData)
        } catch {
            throw LLMError.decodingError("Failed to decode JSON: \(error.localizedDescription)")
        }
    }

    public func streamCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) -> AsyncThrowingStream<String, Error> {
        let request = AnthropicAPIClient.MessagesRequest(
            model: options.model ?? defaultModel,
            messages: [
                .init(role: "user", content: prompt)
            ],
            system: systemPrompt,
            max_tokens: options.maxTokens ?? 4096,
            temperature: options.temperature,
            top_p: options.topP,
            stream: true
        )

        return client.streamMessage(request: request)
    }

    public func estimateTokens(_ text: String) async throws -> Int {
        await client.estimateTokens(text)
    }
}

// MARK: - Convenience Initializers

extension AnthropicProvider {
    // MARK: Claude 4.5 Models (Latest - November 2025)

    /// Create provider for Claude Opus 4.5 (Latest flagship model - November 2025)
    /// Most intelligent model, best for coding, agents, and computer use
    public static func opus45(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-opus-4-5-20251124")
    }

    /// Create provider for Claude Sonnet 4.5 (September 2025)
    /// Best coding model with substantial gains in reasoning and math
    public static func sonnet45(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-sonnet-4-5-20250929")
    }

    /// Create provider for Claude Haiku 4.5 (October 2025)
    /// Small, fast model optimized for low latency and cost
    public static func haiku45(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-haiku-4-5-20251015")
    }

    // MARK: Claude 4.x Models

    /// Create provider for Claude Opus 4.1 (August 2025)
    /// Focused on agentic tasks, real-world coding, and reasoning
    public static func opus41(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-opus-4-1-20250805")
    }

    /// Create provider for Claude Sonnet 4 (May 2025)
    public static func sonnet4(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-sonnet-4-20250522")
    }

    /// Create provider for Claude Opus 4 (May 2025)
    public static func opus4(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-opus-4-20250522")
    }

    // MARK: Claude 3.x Models (Legacy)

    /// Create provider for Claude 3.5 Sonnet (Legacy)
    public static func sonnet35(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-3-5-sonnet-20241022")
    }

    /// Create provider for Claude 3 Opus (Legacy)
    public static func opus3(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-3-opus-20240229")
    }

    /// Create provider for Claude 3 Haiku (Legacy)
    public static func haiku3(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-3-haiku-20240307")
    }
}
