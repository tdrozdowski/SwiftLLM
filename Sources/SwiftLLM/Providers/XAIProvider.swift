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
            maxContextTokens: modelContextWindow,
            maxOutputTokens: 4096,
            supportsSystemPrompts: true,
            pricing: modelPricing
        )
    }

    private var modelContextWindow: Int {
        // Grok 4.1 Fast has 2M context window
        if defaultModel.contains("grok-4-1-fast") {
            return 2_000_000
        } else if defaultModel.contains("grok-4") {
            return 500_000 // Grok 4.1 standard models
        } else if defaultModel.contains("grok-2") {
            return 131_072
        } else {
            return 131_072
        }
    }

    private var modelPricing: LLMPricing? {
        // Grok 4.1 Fast pricing (as of November 2025)
        if defaultModel.contains("grok-4-1-fast") {
            return LLMPricing(inputCostPer1M: 0.20, outputCostPer1M: 0.50)
        } else if defaultModel.contains("grok-4-1") {
            return LLMPricing(inputCostPer1M: 2.0, outputCostPer1M: 10.0) // Estimated
        } else if defaultModel.contains("grok-2") {
            return LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 15.0)
        }
        return nil
    }

    public init(apiKey: String, model: String = "grok-4-1-fast-non-reasoning") {
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
        // Build messages
        var messages: [XAIAPIClient.ChatCompletionRequest.Message] = []

        if let system = systemPrompt {
            messages.append(.init(role: "system", content: system))
        }
        messages.append(.init(role: "user", content: prompt))

        // Use response_format to force JSON mode (no markdown wrapping)
        let request = XAIAPIClient.ChatCompletionRequest(
            model: options.model ?? defaultModel,
            messages: messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            stream: false,
            response_format: XAIAPIClient.ChatCompletionRequest.ResponseFormat(type: "json_object")
        )

        let response = try await client.createChatCompletion(request: request)

        guard let choice = response.choices.first,
              let content = choice.message.content else {
            throw LLMError.decodingError("No content in response")
        }

        // Strip markdown code blocks if present (```json ... ``` or ``` ... ```)
        let cleanedContent = Self.stripMarkdownCodeBlocks(content)

        guard let jsonData = cleanedContent.data(using: .utf8) else {
            throw LLMError.decodingError("Failed to convert response to data")
        }

        do {
            return try JSONDecoder().decode(T.self, from: jsonData)
        } catch {
            throw LLMError.decodingError("Failed to decode JSON: \(error.localizedDescription)\nResponse: \(cleanedContent)")
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
            stream: true,
            response_format: nil
        )

        return client.streamChatCompletion(request: request)
    }

    public func estimateTokens(_ text: String) async throws -> Int {
        await client.estimateTokens(text)
    }

    // MARK: - Private Helpers

    /// Strip markdown code blocks from LLM response
    /// Handles ```json ... ```, ``` ... ```, and plain JSON
    private static func stripMarkdownCodeBlocks(_ content: String) -> String {
        var result = content.trimmingCharacters(in: .whitespacesAndNewlines)

        // Check for ```json or ``` at the start
        if result.hasPrefix("```json") {
            result = String(result.dropFirst(7))
        } else if result.hasPrefix("```") {
            result = String(result.dropFirst(3))
        }

        // Check for ``` at the end
        if result.hasSuffix("```") {
            result = String(result.dropLast(3))
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Convenience Initializers

extension XAIProvider {
    // MARK: Grok 4.1 Models (Latest - November 2025)

    /// Create provider for Grok 4.1 Fast - Non-Reasoning (November 2025)
    /// Best tool-calling model with 2M context window
    /// Optimized for speed, accuracy, and enterprise applications
    /// Ranks #2 on LMArena at 1465 Elo
    public static func grok41FastNonReasoning(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-4-1-fast-non-reasoning")
    }

    /// Create provider for Grok 4.1 Fast - Reasoning (November 2025)
    /// Advanced reasoning with 2M context window
    /// Long-horizon reinforcement learning with multi-turn emphasis
    public static func grok41FastReasoning(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-4-1-fast-reasoning")
    }

    /// Create provider for Grok 4.1 Thinking (November 2025)
    /// Top-ranked model on LMArena with 1483 Elo (#1 overall)
    /// Exceptional in creative, emotional, and collaborative interactions
    /// 3x less likely to hallucinate than previous models
    public static func grok41Thinking(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-4-1-thinking")
    }

    /// Create provider for Grok 4.1 (November 2025)
    /// Immediate response with no thinking tokens
    /// Ranks #2 on LMArena at 1465 Elo
    public static func grok41(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-4-1")
    }

    // MARK: Grok 2 Models (Legacy)

    /// Create provider for Grok-2 (Legacy)
    public static func grok2(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-2-latest")
    }

    /// Create provider for Grok-2 Mini (Legacy)
    public static func grok2Mini(apiKey: String) -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: "grok-2-mini-latest")
    }
}
