import Foundation

/// Registry for managing and creating LLM providers
public final class ProviderRegistry: @unchecked Sendable {
    /// Shared singleton instance
    public static let shared = ProviderRegistry()

    /// Provider configuration
    public struct ProviderConfig: Sendable {
        public let type: ProviderType
        public let apiKey: String
        public let model: String?
        public let baseURL: String?

        public init(
            type: ProviderType,
            apiKey: String = "",
            model: String? = nil,
            baseURL: String? = nil
        ) {
            self.type = type
            self.apiKey = apiKey
            self.model = model
            self.baseURL = baseURL
        }
    }

    /// Supported provider types
    public enum ProviderType: String, Sendable, CaseIterable {
        case anthropic
        case openai
        case xai
        case ollama
        case lmstudio
        case localai
        case openaiCompatible
    }

    private let lock = NSLock()
    private var providers: [String: any LLMProvider] = [:]
    private var configs: [String: ProviderConfig] = [:]

    public init() {}

    /// Register a provider configuration
    public func register(_ config: ProviderConfig, as name: String) {
        lock.lock()
        defer { lock.unlock() }
        configs[name] = config
    }

    /// Register a pre-configured provider instance
    public func register(_ provider: any LLMProvider, as name: String) {
        lock.lock()
        defer { lock.unlock() }
        providers[name] = provider
    }

    /// Get a provider by name, creating it if necessary from config
    public func provider(named name: String) -> (any LLMProvider)? {
        lock.lock()
        defer { lock.unlock() }

        // Return cached provider if available
        if let provider = providers[name] {
            return provider
        }

        // Try to create from config
        guard let config = configs[name] else {
            return nil
        }

        let provider = createProvider(from: config)
        providers[name] = provider
        return provider
    }

    /// Get or create an Anthropic provider
    public func anthropic(apiKey: String, model: String = "claude-sonnet-4-5-20250514") -> AnthropicProvider {
        AnthropicProvider(apiKey: apiKey, model: model)
    }

    /// Get or create an OpenAI provider
    public func openai(apiKey: String, model: String = "gpt-5-1-instant") -> OpenAIProvider {
        OpenAIProvider(apiKey: apiKey, model: model)
    }

    /// Get or create an xAI provider
    public func xai(apiKey: String, model: String = "grok-4-1") -> XAIProvider {
        XAIProvider(apiKey: apiKey, model: model)
    }

    /// Get or create an Ollama provider
    public func ollama(model: String, host: String = "localhost", port: Int = 11434) -> LocalLLMProvider {
        LocalLLMProvider.ollama(model: model, host: host, port: port)
    }

    /// Get or create an LM Studio provider
    public func lmStudio(model: String, host: String = "localhost", port: Int = 1234) -> LocalLLMProvider {
        LocalLLMProvider.lmStudio(model: model, host: host, port: port)
    }

    /// List all registered provider names
    public func registeredNames() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(Set(providers.keys).union(configs.keys)).sorted()
    }

    /// List all available provider types
    public func availableTypes() -> [ProviderType] {
        ProviderType.allCases
    }

    /// Remove a registered provider
    public func unregister(named name: String) {
        lock.lock()
        defer { lock.unlock() }
        providers.removeValue(forKey: name)
        configs.removeValue(forKey: name)
    }

    /// Clear all registered providers
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        providers.removeAll()
        configs.removeAll()
    }

    // MARK: - Private

    private func createProvider(from config: ProviderConfig) -> any LLMProvider {
        switch config.type {
        case .anthropic:
            return AnthropicProvider(apiKey: config.apiKey, model: config.model ?? "claude-sonnet-4-5-20250514")

        case .openai:
            return OpenAIProvider(apiKey: config.apiKey, model: config.model ?? "gpt-5-1-instant")

        case .xai:
            return XAIProvider(apiKey: config.apiKey, model: config.model ?? "grok-4-1")

        case .ollama:
            return LocalLLMProvider.ollama(model: config.model ?? "llama3.2")

        case .lmstudio:
            return LocalLLMProvider.lmStudio(model: config.model ?? "default")

        case .localai:
            return LocalLLMProvider.localAI(model: config.model ?? "default")

        case .openaiCompatible:
            return LocalLLMProvider.openAICompatible(
                baseURL: config.baseURL ?? "http://localhost:8000",
                model: config.model ?? "default",
                apiKey: config.apiKey
            )
        }
    }
}

// MARK: - Environment-based Configuration

extension ProviderRegistry {
    /// Configure providers from environment variables
    /// Looks for: ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY
    public func configureFromEnvironment() {
        let env = ProcessInfo.processInfo.environment

        if let anthropicKey = env["ANTHROPIC_API_KEY"], !anthropicKey.isEmpty {
            register(ProviderConfig(type: .anthropic, apiKey: anthropicKey), as: "anthropic")
        }

        if let openaiKey = env["OPENAI_API_KEY"], !openaiKey.isEmpty {
            register(ProviderConfig(type: .openai, apiKey: openaiKey), as: "openai")
        }

        if let xaiKey = env["XAI_API_KEY"], !xaiKey.isEmpty {
            register(ProviderConfig(type: .xai, apiKey: xaiKey), as: "xai")
        }

        // Local providers don't need API keys
        register(ProviderConfig(type: .ollama), as: "ollama")
        register(ProviderConfig(type: .lmstudio), as: "lmstudio")
    }
}
