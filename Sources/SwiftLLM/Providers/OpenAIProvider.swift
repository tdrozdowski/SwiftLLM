import Foundation

/// Provider implementation for OpenAI GPT models
public struct OpenAIProvider: LLMProvider {
    public let id: String = "openai"
    public let displayName: String
    private let client: OpenAIAPIClient
    private let defaultModel: String

    public var capabilities: LLMCapabilities {
        LLMCapabilities(
            supportsStructuredOutput: modelSupportsStructuredOutput,
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

    private var modelSupportsStructuredOutput: Bool {
        defaultModel.contains("gpt-4") || defaultModel.contains("gpt-3.5-turbo")
    }

    private var modelSupportsVision: Bool {
        defaultModel.contains("gpt-4-vision") || defaultModel.contains("gpt-4o")
    }

    private var modelContextWindow: Int {
        if defaultModel.contains("gpt-4o") {
            return 128_000
        } else if defaultModel.contains("gpt-4-turbo") {
            return 128_000
        } else if defaultModel.contains("gpt-4") {
            return 8192
        } else if defaultModel.contains("gpt-3.5-turbo-16k") {
            return 16_384
        } else {
            return 4096
        }
    }

    private var modelMaxOutput: Int {
        if defaultModel.contains("gpt-4o") {
            return 16_384
        } else if defaultModel.contains("gpt-4-turbo") {
            return 4096
        } else {
            return 4096
        }
    }

    private var modelPricing: LLMPricing? {
        // GPT-4o pricing (as of 2024)
        if defaultModel.contains("gpt-4o") {
            return LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("gpt-4-turbo") {
            return LLMPricing(inputCostPer1M: 10.0, outputCostPer1M: 30.0)
        } else if defaultModel.contains("gpt-4") {
            return LLMPricing(inputCostPer1M: 30.0, outputCostPer1M: 60.0)
        } else if defaultModel.contains("gpt-3.5-turbo") {
            return LLMPricing(inputCostPer1M: 0.5, outputCostPer1M: 1.5)
        }
        return nil
    }

    public init(apiKey: String, model: String = "gpt-4o") {
        self.client = OpenAIAPIClient(apiKey: apiKey)
        self.defaultModel = model
        self.displayName = "OpenAI GPT"
    }

    public func generateCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) async throws -> CompletionResponse {
        var messages: [OpenAIAPIClient.ChatCompletionRequest.Message] = []

        if let system = systemPrompt {
            messages.append(.init(role: "system", content: system))
        }
        messages.append(.init(role: "user", content: prompt))

        let request = OpenAIAPIClient.ChatCompletionRequest(
            model: options.model ?? defaultModel,
            messages: messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            frequency_penalty: options.frequencyPenalty,
            presence_penalty: options.presencePenalty,
            stop: options.stopSequences,
            stream: false,
            response_format: nil
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
        var messages: [OpenAIAPIClient.ChatCompletionRequest.Message] = []

        if let system = systemPrompt {
            messages.append(.init(role: "system", content: system))
        }
        messages.append(.init(role: "user", content: prompt))

        let request = OpenAIAPIClient.ChatCompletionRequest(
            model: options.model ?? defaultModel,
            messages: messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            frequency_penalty: options.frequencyPenalty,
            presence_penalty: options.presencePenalty,
            stop: options.stopSequences,
            stream: false,
            response_format: .init(type: "json_object")
        )

        let response = try await client.createChatCompletion(request: request)

        guard let choice = response.choices.first,
              let content = choice.message.content,
              let jsonData = content.data(using: .utf8) else {
            throw LLMError.decodingError("No content in response")
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
        var messages: [OpenAIAPIClient.ChatCompletionRequest.Message] = []

        if let system = systemPrompt {
            messages.append(.init(role: "system", content: system))
        }
        messages.append(.init(role: "user", content: prompt))

        let request = OpenAIAPIClient.ChatCompletionRequest(
            model: options.model ?? defaultModel,
            messages: messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            frequency_penalty: options.frequencyPenalty,
            presence_penalty: options.presencePenalty,
            stop: options.stopSequences,
            stream: true,
            response_format: nil
        )

        return client.streamChatCompletion(request: request)
    }

    public func estimateTokens(_ text: String) async throws -> Int {
        await client.estimateTokens(text)
    }
}

// MARK: - Convenience Initializers

extension OpenAIProvider {
    /// Create provider for GPT-4o
    public static func gpt4o(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-4o")
    }

    /// Create provider for GPT-4 Turbo
    public static func gpt4Turbo(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-4-turbo")
    }

    /// Create provider for GPT-3.5 Turbo
    public static func gpt35Turbo(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-3.5-turbo")
    }
}
