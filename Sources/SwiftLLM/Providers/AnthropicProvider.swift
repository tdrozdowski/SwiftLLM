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
        defaultModel.contains("claude-3")
    }

    private var modelContextWindow: Int {
        if defaultModel.contains("claude-3-5") {
            return 200_000
        } else if defaultModel.contains("claude-3") {
            return 200_000
        } else {
            return 100_000
        }
    }

    private var modelMaxOutput: Int {
        if defaultModel.contains("claude-3-5") {
            return 8192
        } else {
            return 4096
        }
    }

    private var modelPricing: LLMPricing? {
        // Claude 3.5 Sonnet pricing (as of 2024)
        if defaultModel.contains("claude-3-5-sonnet") {
            return LLMPricing(inputCostPer1M: 3.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("claude-3-opus") {
            return LLMPricing(inputCostPer1M: 15.0, outputCostPer1M: 75.0)
        } else if defaultModel.contains("claude-3-sonnet") {
            return LLMPricing(inputCostPer1M: 3.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("claude-3-haiku") {
            return LLMPricing(inputCostPer1M: 0.25, outputCostPer1M: 1.25)
        }
        return nil
    }

    public init(apiKey: String, model: String = "claude-3-5-sonnet-20241022") {
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
    /// Create provider for Claude 3.5 Sonnet
    public static func sonnet(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-3-5-sonnet-20241022")
    }

    /// Create provider for Claude 3 Opus
    public static func opus(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-3-opus-20240229")
    }

    /// Create provider for Claude 3 Haiku
    public static func haiku(apiKey: String) -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: "claude-3-haiku-20240307")
    }
}
