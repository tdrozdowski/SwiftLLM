import Foundation

/// Provider implementation for xAI Grok models
public struct XAIProvider: LLMProvider {
    public let id: String = "xai"
    public let displayName: String
    private let client: XAIAPIClient
    private let defaultModel: String

    public var capabilities: LLMCapabilities {
        LLMCapabilities(
            supportsStructuredOutput: false,
            supportsStreaming: true,
            supportsLocalExecution: false,
            supportsVision: false,
            supportsToolCalling: true,
            maxContextTokens: 131_072, // Grok-2 context window
            maxOutputTokens: 4096,
            supportsSystemPrompts: true,
            pricing: LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 15.0) // Approximate pricing
        )
    }

    public init(apiKey: String, model: String = "grok-2-latest") {
        self.client = XAIAPIClient(apiKey: apiKey)
        self.defaultModel = model
        self.displayName = "xAI Grok"
    }

    public func generateCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) async throws -> CompletionResponse {
        var messages: [XAIAPIClient.ChatCompletionRequest.Message] = []

        if let system = systemPrompt {
            messages.append(.init(role: "system", content: system))
        }
        messages.append(.init(role: "user", content: prompt))

        let request = XAIAPIClient.ChatCompletionRequest(
            model: options.model ?? defaultModel,
            messages: messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            stream: false
        )

        let response = try await client.createChatCompletion(request: request)

        guard let choice = response.choices.first,
              let content = choice.message.content else {
            throw LLMError.decodingError("No content in response")
        }

        return CompletionResponse(
            text: content,
            model: response.model,
            usage: TokenUsage(
                inputTokens: response.usage.prompt_tokens,
                outputTokens: response.usage.completion_tokens
            ),
            finishReason: choice.finish_reason,
            metadata: ["id": response.id, "created": "\(response.created)"]
        )
    }

    public func generateStructuredOutput<T: Codable>(
        prompt: String,
        systemPrompt: String?,
        schema: T.Type,
        options: GenerationOptions
    ) async throws -> T {
        // Use prompt engineering for JSON output
        let jsonPrompt = """
        \(prompt)

        Respond with valid JSON only. No other text or markdown formatting.
        """

        let response = try await generateCompletion(
            prompt: jsonPrompt,
            systemPrompt: systemPrompt,
            options: options
        )

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
        var messages: [XAIAPIClient.ChatCompletionRequest.Message] = []

        if let system = systemPrompt {
            messages.append(.init(role: "system", content: system))
        }
        messages.append(.init(role: "user", content: prompt))

        let request = XAIAPIClient.ChatCompletionRequest(
            model: options.model ?? defaultModel,
            messages: messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            stream: true
        )

        return client.streamChatCompletion(request: request)
    }

    public func estimateTokens(_ text: String) async throws -> Int {
        await client.estimateTokens(text)
    }
}

// MARK: - Convenience Initializers

extension XAIProvider {
    /// Create provider for Grok-2 (latest)
    public static func grok2(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-2-latest")
    }

    /// Create provider for Grok-2 Mini
    public static func grok2Mini(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-2-mini-latest")
    }
}
