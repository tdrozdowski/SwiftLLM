import Foundation

/// Configuration for a local LLM model
public struct LocalModelConfig: Sendable {
    public let name: String
    public let contextWindow: Int
    public let supportsVision: Bool
    public let supportsToolCalling: Bool
    public let supportsStructuredOutput: Bool

    public init(
        name: String,
        contextWindow: Int = 4096,
        supportsVision: Bool = false,
        supportsToolCalling: Bool = false,
        supportsStructuredOutput: Bool = true
    ) {
        self.name = name
        self.contextWindow = contextWindow
        self.supportsVision = supportsVision
        self.supportsToolCalling = supportsToolCalling
        self.supportsStructuredOutput = supportsStructuredOutput
    }
}

/// Provider for local LLM servers with OpenAI-compatible APIs
/// Supports Ollama, LM Studio, LocalAI, llama.cpp server, and similar tools
public struct LocalLLMProvider: LLMProvider {
    public let id: String
    public let displayName: String
    private let client: OpenAIAPIClient
    private let modelConfig: LocalModelConfig

    public var capabilities: LLMCapabilities {
        LLMCapabilities(
            supportsStructuredOutput: modelConfig.supportsStructuredOutput,
            supportsStreaming: true,
            supportsLocalExecution: true,
            supportsVision: modelConfig.supportsVision,
            supportsToolCalling: modelConfig.supportsToolCalling,
            maxContextTokens: modelConfig.contextWindow,
            maxOutputTokens: modelConfig.contextWindow / 2,
            supportsSystemPrompts: true,
            pricing: nil // Local = free
        )
    }

    /// Create a provider for a local OpenAI-compatible LLM server
    /// - Parameters:
    ///   - baseURL: The base URL of the local server (e.g., "http://localhost:11434" for Ollama)
    ///   - model: Configuration for the model to use
    ///   - apiKey: Optional API key (some servers don't require one)
    ///   - displayName: Display name for this provider
    public init(
        baseURL: URL,
        model: LocalModelConfig,
        apiKey: String = "",
        displayName: String? = nil
    ) {
        self.client = OpenAIAPIClient(apiKey: apiKey, baseURL: baseURL)
        self.modelConfig = model
        self.displayName = displayName ?? "Local LLM (\(model.name))"
        self.id = "local-\(model.name)"
    }

    /// Convenience initializer with just model name (uses defaults)
    public init(
        baseURL: URL,
        modelName: String,
        apiKey: String = "",
        displayName: String? = nil
    ) {
        self.init(
            baseURL: baseURL,
            model: LocalModelConfig(name: modelName),
            apiKey: apiKey,
            displayName: displayName
        )
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
            model: options.model ?? modelConfig.name,
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
            model: options.model ?? modelConfig.name,
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
            model: options.model ?? modelConfig.name,
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
        // Rough approximation - varies by model
        return max(1, text.count / 4)
    }
}

// MARK: - Convenience Initializers for Popular Local LLM Servers

extension LocalLLMProvider {

    // MARK: Ollama

    /// Create provider for Ollama with any model
    /// - Parameters:
    ///   - model: Model configuration (use LocalModelConfig for full control)
    ///   - host: Host address (default: "localhost")
    ///   - port: Port number (default: 11434)
    public static func ollama(
        model: LocalModelConfig,
        host: String = "localhost",
        port: Int = 11434
    ) -> LocalLLMProvider {
        LocalLLMProvider(
            baseURL: URL(string: "http://\(host):\(port)")!,
            model: model,
            displayName: "Ollama (\(model.name))"
        )
    }

    /// Create provider for Ollama with just a model name (uses defaults)
    public static func ollama(
        model: String,
        host: String = "localhost",
        port: Int = 11434
    ) -> LocalLLMProvider {
        ollama(model: LocalModelConfig(name: model), host: host, port: port)
    }

    // MARK: LM Studio

    /// Create provider for LM Studio with any model
    /// - Parameters:
    ///   - model: Model configuration
    ///   - host: Host address (default: "localhost")
    ///   - port: Port number (default: 1234)
    public static func lmStudio(
        model: LocalModelConfig,
        host: String = "localhost",
        port: Int = 1234
    ) -> LocalLLMProvider {
        LocalLLMProvider(
            baseURL: URL(string: "http://\(host):\(port)")!,
            model: model,
            displayName: "LM Studio (\(model.name))"
        )
    }

    /// Create provider for LM Studio with just a model name
    public static func lmStudio(
        model: String,
        host: String = "localhost",
        port: Int = 1234
    ) -> LocalLLMProvider {
        lmStudio(model: LocalModelConfig(name: model), host: host, port: port)
    }

    // MARK: LocalAI

    /// Create provider for LocalAI with any model
    public static func localAI(
        model: LocalModelConfig,
        host: String = "localhost",
        port: Int = 8080
    ) -> LocalLLMProvider {
        LocalLLMProvider(
            baseURL: URL(string: "http://\(host):\(port)")!,
            model: model,
            displayName: "LocalAI (\(model.name))"
        )
    }

    /// Create provider for LocalAI with just a model name
    public static func localAI(
        model: String,
        host: String = "localhost",
        port: Int = 8080
    ) -> LocalLLMProvider {
        localAI(model: LocalModelConfig(name: model), host: host, port: port)
    }

    // MARK: llama.cpp Server

    /// Create provider for llama.cpp server
    public static func llamaCpp(
        model: LocalModelConfig = LocalModelConfig(name: "default"),
        host: String = "localhost",
        port: Int = 8080
    ) -> LocalLLMProvider {
        LocalLLMProvider(
            baseURL: URL(string: "http://\(host):\(port)")!,
            model: model,
            displayName: "llama.cpp (\(model.name))"
        )
    }

    // MARK: Generic OpenAI-Compatible

    /// Create provider for any OpenAI-compatible server
    public static func openAICompatible(
        baseURL: String,
        model: LocalModelConfig,
        apiKey: String = "",
        displayName: String? = nil
    ) -> LocalLLMProvider {
        LocalLLMProvider(
            baseURL: URL(string: baseURL)!,
            model: model,
            apiKey: apiKey,
            displayName: displayName
        )
    }

    /// Create provider for any OpenAI-compatible server with just a model name
    public static func openAICompatible(
        baseURL: String,
        model: String,
        apiKey: String = "",
        displayName: String? = nil
    ) -> LocalLLMProvider {
        openAICompatible(
            baseURL: baseURL,
            model: LocalModelConfig(name: model),
            apiKey: apiKey,
            displayName: displayName
        )
    }
}
