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
        // GPT-5 and GPT-4 models support structured output
        defaultModel.contains("gpt-5") || defaultModel.contains("gpt-4") || defaultModel.contains("gpt-3.5-turbo")
    }

    private var modelSupportsVision: Bool {
        // GPT-5 and GPT-4o models support vision
        defaultModel.contains("gpt-5") || defaultModel.contains("gpt-4-vision") || defaultModel.contains("gpt-4o")
    }

    private var modelContextWindow: Int {
        // GPT-5.1-Codex-Max can work over millions of tokens through compaction
        if defaultModel.contains("gpt-5-1-codex-max") {
            return 1_000_000 // Can handle millions through compaction
        } else if defaultModel.contains("gpt-5") {
            return 200_000
        } else if defaultModel.contains("gpt-4o") || defaultModel.contains("gpt-4-turbo") {
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
        if defaultModel.contains("gpt-5") || defaultModel.contains("gpt-4o") {
            return 16_384
        } else if defaultModel.contains("gpt-4-turbo") {
            return 4096
        } else {
            return 4096
        }
    }

    private var modelPricing: LLMPricing? {
        // GPT-5.1 pricing (as of November 2025) - estimated based on GPT-4o
        if defaultModel.contains("gpt-5-1-codex-max") {
            return LLMPricing(inputCostPer1M: 10.0, outputCostPer1M: 30.0)
        } else if defaultModel.contains("gpt-5-1-codex") {
            return LLMPricing(inputCostPer1M: 7.0, outputCostPer1M: 20.0)
        } else if defaultModel.contains("gpt-5-1") {
            return LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("gpt-5") {
            return LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 15.0)
        } else if defaultModel.contains("gpt-4o") {
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

    public init(apiKey: String, model: String = "gpt-5-1-instant") {
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

    // MARK: - Tool Calling Support

    public func generateCompletionWithTools(
        context: ConversationContext,
        toolChoice: ToolChoice = .auto,
        options: GenerationOptions = .default
    ) async throws -> CompletionResponse {
        let messages = try convertContextToMessages(context)
        let tools = convertTools(context.tools)
        let toolChoiceValue = convertToolChoice(toolChoice)

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
            response_format: nil,
            tools: tools,
            tool_choice: toolChoiceValue
        )

        let response = try await client.createChatCompletion(request: request)
        return try buildCompletionResponse(from: response)
    }

    public func continueWithToolResults(
        context: ConversationContext,
        options: GenerationOptions = .default
    ) async throws -> CompletionResponse {
        // Same as generateCompletionWithTools - context contains all messages including tool results
        let messages = try convertContextToMessages(context)
        let tools = convertTools(context.tools)

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
            response_format: nil,
            tools: tools,
            tool_choice: nil // Let model decide
        )

        let response = try await client.createChatCompletion(request: request)
        return try buildCompletionResponse(from: response)
    }

    // MARK: - Helper Methods

    private func buildCompletionResponse(from response: OpenAIAPIClient.ChatCompletionResponse) throws -> CompletionResponse {
        guard let choice = response.choices.first else {
            throw LLMError.decodingError("No choices in response")
        }

        // Extract tool calls if present
        let toolCalls = choice.message.tool_calls?.map { openAICall in
            ToolCall(
                id: openAICall.id,
                name: openAICall.function.name,
                arguments: openAICall.function.arguments
            )
        } ?? []

        return CompletionResponse(
            text: choice.message.content ?? "",
            model: response.model,
            usage: TokenUsage(
                inputTokens: response.usage.prompt_tokens,
                outputTokens: response.usage.completion_tokens
            ),
            finishReason: choice.finish_reason,
            metadata: ["id": response.id, "created": "\(response.created)"],
            toolCalls: toolCalls
        )
    }

    private func convertContextToMessages(_ context: ConversationContext) throws -> [OpenAIAPIClient.ChatCompletionRequest.Message] {
        var messages: [OpenAIAPIClient.ChatCompletionRequest.Message] = []

        // Add system prompt if present
        if let systemPrompt = context.systemPrompt {
            messages.append(.init(role: "system", content: systemPrompt))
        }

        // Convert all conversation messages
        for message in context.messages {
            switch message.role {
            case .user:
                messages.append(.init(role: "user", content: message.content))

            case .assistant:
                // Convert tool calls if present
                let openAIToolCalls = message.toolCalls?.map { toolCall in
                    OpenAIAPIClient.ChatCompletionRequest.OpenAIToolCall(
                        id: toolCall.id,
                        type: "function",
                        function: .init(name: toolCall.name, arguments: toolCall.arguments)
                    )
                }
                messages.append(.init(
                    role: "assistant",
                    content: message.content,
                    tool_calls: openAIToolCalls
                ))

            case .tool:
                // OpenAI expects one message per tool result
                if let toolResults = message.toolResults {
                    for result in toolResults {
                        let content: String
                        switch result.result {
                        case .success(let text):
                            content = text
                        case .error(let error):
                            content = error.toLLMFormat()
                        }
                        messages.append(.init(
                            role: "tool",
                            content: content,
                            tool_call_id: result.toolCallId
                        ))
                    }
                }

            case .system:
                messages.append(.init(role: "system", content: message.content))
            }
        }

        return messages
    }

    private func convertTools(_ tools: [Tool]) -> [OpenAIAPIClient.ChatCompletionRequest.OpenAITool] {
        tools.map { tool in
            // Convert ToolParameters to SendableValue
            let parametersValue = convertToolParametersToValue(tool.parameters)

            return OpenAIAPIClient.ChatCompletionRequest.OpenAITool(
                type: "function",
                function: .init(
                    name: tool.name,
                    description: tool.description,
                    parameters: OpenAIAPIClient.AnyCodable(parametersValue)
                )
            )
        }
    }

    private func convertToolParametersToValue(_ parameters: ToolParameters) -> OpenAIAPIClient.AnyCodable.SendableValue {
        var dict: [String: OpenAIAPIClient.AnyCodable.SendableValue] = [
            "type": .string(parameters.type)
        ]

        // Convert properties
        var propertiesDict: [String: OpenAIAPIClient.AnyCodable.SendableValue] = [:]
        for (key, property) in parameters.properties {
            propertiesDict[key] = convertToolPropertyToValue(property)
        }
        dict["properties"] = .dictionary(propertiesDict)

        // Add required if present
        if let required = parameters.required {
            dict["required"] = .array(required.map { .string($0) })
        }

        return .dictionary(dict)
    }

    private func convertToolPropertyToValue(_ property: ToolProperty) -> OpenAIAPIClient.AnyCodable.SendableValue {
        var dict: [String: OpenAIAPIClient.AnyCodable.SendableValue] = [
            "type": .string(property.type)
        ]

        if let description = property.description {
            dict["description"] = .string(description)
        }

        // Handle enum for string types
        if case .string(_, let enumValues) = property, let enumValues = enumValues {
            dict["enum"] = .array(enumValues.map { .string($0) })
        }

        // Handle array items
        if let items = property.items {
            dict["items"] = convertToolPropertyToValue(items)
        }

        // Handle object properties
        if let properties = property.properties {
            var propertiesDict: [String: OpenAIAPIClient.AnyCodable.SendableValue] = [:]
            for (key, prop) in properties {
                propertiesDict[key] = convertToolPropertyToValue(prop)
            }
            dict["properties"] = .dictionary(propertiesDict)
        }

        return .dictionary(dict)
    }

    private func convertToolChoice(_ choice: ToolChoice) -> OpenAIToolChoice {
        choice.toOpenAIFormat()
    }
}

// MARK: - Convenience Initializers

extension OpenAIProvider {
    // MARK: GPT-5.1 Models (Latest - November 2025)

    /// Create provider for GPT-5.1-Codex-Max (November 2025)
    /// Frontier agentic coding model built for long-running, detailed work
    /// First model natively trained to operate across multiple context windows (millions of tokens)
    public static func gpt51CodexMax(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-5-1-codex-max")
    }

    /// Create provider for GPT-5.1-Codex
    /// Optimized for coding tasks with enhanced tool-calling
    public static func gpt51Codex(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-5-1-codex")
    }

    /// Create provider for GPT-5.1-Codex-Mini
    /// Smaller, faster coding model
    public static func gpt51CodexMini(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-5-1-codex-mini")
    }

    /// Create provider for GPT-5.1 Instant (November 2025)
    /// Most-used model - warmer, more intelligent, better at following instructions
    /// Adaptive reasoning with "no reasoning" mode for faster responses
    public static func gpt51Instant(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-5-1-instant")
    }

    /// Create provider for GPT-5.1 Thinking (November 2025)
    /// Advanced reasoning model, easier to understand and faster on simple tasks
    /// More persistent on complex problems
    public static func gpt51Thinking(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-5-1-thinking")
    }

    // MARK: GPT-5 Models (August 2025)

    /// Create provider for GPT-5 (August 2025)
    /// Major frontier model update with state-of-the-art performance
    public static func gpt5(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-5")
    }

    // MARK: GPT-4 Models (Legacy)

    /// Create provider for GPT-4o (Legacy)
    public static func gpt4o(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-4o")
    }

    /// Create provider for GPT-4 Turbo (Legacy)
    public static func gpt4Turbo(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-4-turbo")
    }

    /// Create provider for GPT-3.5 Turbo (Legacy)
    public static func gpt35Turbo(apiKey: String) -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: "gpt-3.5-turbo")
    }
}
