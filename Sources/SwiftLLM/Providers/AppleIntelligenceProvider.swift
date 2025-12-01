#if canImport(FoundationModels)
import Foundation
import FoundationModels
import os.log

/// Provider implementation for Apple Foundation Models (AFM)
/// Uses the official FoundationModels framework from macOS 26+
/// Supports on-device and server-based Apple Intelligence models
@available(macOS 26.0, iOS 26.0, *)
public struct AppleIntelligenceProvider: LLMProvider {
    public let id: String = "apple"
    public let displayName: String
    private let instructions: String?
    private let modelType: AFMModelType

    public enum AFMModelType: Sendable {
        case onDevice    // 3B parameter on-device model
        case server      // Larger server-hosted mixture-of-experts model

        var displayName: String {
            switch self {
            case .onDevice: return "Apple Intelligence (On-Device)"
            case .server: return "Apple Intelligence (Server)"
            }
        }
    }

    public var capabilities: LLMCapabilities {
        LLMCapabilities(
            supportsStructuredOutput: true,  // AFM supports structured output via Generable protocol
            supportsStreaming: true,         // AFM supports streaming via ResponseStream
            supportsLocalExecution: modelType == .onDevice,
            supportsVision: true,            // AFM includes vision transformer (300M on-device, 1B server)
            supportsToolCalling: true,       // AFM supports tool calling
            maxContextTokens: modelType == .onDevice ? 8192 : 32_768,
            maxOutputTokens: 4096,
            supportsSystemPrompts: true,
            pricing: nil // On-device is free; server pricing TBD
        )
    }

    public init(modelType: AFMModelType = .onDevice, instructions: String? = nil) {
        self.modelType = modelType
        self.instructions = instructions
        self.displayName = modelType.displayName
    }

    public func generateCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) async throws -> CompletionResponse {
        let modelName = modelType == .onDevice ? "afm-on-device" : "afm-server"

        SwiftLLMLogger.provider.info("→ AFM generateCompletion (\(modelName, privacy: .public))")
        SwiftLLMLogger.provider.debug("  Prompt: \(prompt, privacy: .public)")
        if let system = systemPrompt ?? instructions {
            SwiftLLMLogger.provider.debug("  System: \(system, privacy: .public)")
        }

        // Create session with system instructions
        let session: LanguageModelSession
        if let system = systemPrompt ?? instructions {
            session = LanguageModelSession(instructions: system)
        } else {
            session = LanguageModelSession()
        }

        // Convert our GenerationOptions to AFM's GenerationOptions
        let genOptions = FoundationModels.GenerationOptions(
            sampling: nil,
            temperature: options.temperature,
            maximumResponseTokens: options.maxTokens
        )

        do {
            // Generate response
            let response = try await session.respond(options: genOptions) {
                prompt
            }

            let inputTokens = estimateTokensSync(prompt)
            let outputTokens = estimateTokensSync(response.content)

            SwiftLLMLogger.provider.info("← AFM response (\(modelName, privacy: .public))")
            SwiftLLMLogger.provider.debug("  Response: \(response.content, privacy: .public)")
            SwiftLLMLogger.provider.debug("  Tokens: \(inputTokens, privacy: .public) in, \(outputTokens, privacy: .public) out")

            return CompletionResponse(
                text: response.content,
                model: modelName,
                usage: TokenUsage(
                    inputTokens: inputTokens,
                    outputTokens: outputTokens
                ),
                finishReason: nil,
                metadata: [
                    "model_type": modelType == .onDevice ? "on-device" : "server",
                    "on_device": "\(modelType == .onDevice)"
                ]
            )
        } catch {
            SwiftLLMLogger.error.logError(error, context: "AFM generateCompletion")
            throw error
        }
    }

    public func generateStructuredOutput<T: Codable>(
        prompt: String,
        systemPrompt: String?,
        schema: T.Type,
        options: GenerationOptions
    ) async throws -> T {
        let modelName = modelType == .onDevice ? "afm-on-device" : "afm-server"

        SwiftLLMLogger.provider.info("→ AFM generateStructuredOutput (\(modelName, privacy: .public))")
        SwiftLLMLogger.provider.debug("  Prompt: \(prompt, privacy: .public)")
        SwiftLLMLogger.provider.debug("  Schema: \(String(describing: T.self), privacy: .public)")

        // Create session with system instructions
        let session: LanguageModelSession
        if let system = systemPrompt ?? instructions {
            session = LanguageModelSession(instructions: system)
        } else {
            session = LanguageModelSession()
        }

        // Convert our GenerationOptions to AFM's GenerationOptions
        let genOptions = FoundationModels.GenerationOptions(
            sampling: nil,
            temperature: options.temperature,
            maximumResponseTokens: options.maxTokens
        )

        // For structured output, we need to use a Generable-conforming type
        // Since T is Codable but not necessarily Generable, we'll use JSON approach
        let jsonPrompt = """
        \(prompt)

        Respond with valid JSON only. No other text or markdown formatting.
        """

        do {
            let response = try await session.respond(options: genOptions) {
                jsonPrompt
            }

            SwiftLLMLogger.provider.debug("  JSON Response: \(response.content, privacy: .public)")

            guard let jsonData = response.content.data(using: .utf8) else {
                throw LLMError.decodingError("Could not convert response to data")
            }

            let decoded = try JSONDecoder().decode(T.self, from: jsonData)
            SwiftLLMLogger.provider.info("← AFM structured output decoded successfully")
            return decoded
        } catch let error as LLMError {
            SwiftLLMLogger.error.logError(error, context: "AFM generateStructuredOutput")
            throw error
        } catch {
            let decodingError = LLMError.decodingError("Failed to decode JSON: \(error.localizedDescription)")
            SwiftLLMLogger.error.logError(decodingError, context: "AFM generateStructuredOutput")
            throw decodingError
        }
    }

    /// Generate a response with native Apple @Generable type (recommended for AFM)
    ///
    /// This method uses Apple's native guided generation to constrain the model's output
    /// to match the schema of a @Generable type. This is more reliable than JSON prompting
    /// and returns native Swift objects without any parsing.
    ///
    /// - Parameters:
    ///   - prompt: The user's prompt
    ///   - systemPrompt: Optional system instructions (overrides provider-level instructions)
    ///   - responseType: The @Generable type to generate
    ///   - options: Generation options (temperature, max tokens, etc.)
    /// - Returns: The generated response as the specified @Generable type
    /// - Throws: LLMError if generation fails
    ///
    /// Example:
    /// ```swift
    /// @Generable
    /// struct Summary {
    ///     @Guide(description: "Brief one-line summary")
    ///     var brief: String
    ///
    ///     @Guide(description: "Key points")
    ///     var keyPoints: [String]
    /// }
    ///
    /// let provider = AppleIntelligenceProvider.onDevice()
    /// let summary = try await provider.generateGenerable(
    ///     prompt: "Summarize this code: ...",
    ///     systemPrompt: "You are a code analyzer",
    ///     responseType: Summary.self,
    ///     options: .default
    /// )
    /// print(summary.brief)
    /// ```
    public func generateGenerable<T: Generable>(
        prompt: String,
        systemPrompt: String?,
        responseType: T.Type,
        options: GenerationOptions
    ) async throws -> T {
        let modelName = modelType == .onDevice ? "afm-on-device" : "afm-server"

        SwiftLLMLogger.provider.info("→ AFM generateGenerable (\(modelName, privacy: .public))")
        SwiftLLMLogger.provider.debug("  Prompt: \(prompt, privacy: .public)")
        SwiftLLMLogger.provider.debug("  Response Type: \(String(describing: T.self), privacy: .public)")
        if let system = systemPrompt ?? instructions {
            SwiftLLMLogger.provider.debug("  System: \(system, privacy: .public)")
        }

        // Create session with system instructions
        let session: LanguageModelSession
        if let system = systemPrompt ?? instructions {
            session = LanguageModelSession(instructions: system)
        } else {
            session = LanguageModelSession()
        }

        // Convert our GenerationOptions to AFM's GenerationOptions
        let genOptions = FoundationModels.GenerationOptions(
            sampling: nil,
            temperature: options.temperature,
            maximumResponseTokens: options.maxTokens
        )

        do {
            // Use AFM's native guided generation with @Generable type
            let response = try await session.respond(
                to: prompt,
                generating: T.self,
                options: genOptions
            )

            SwiftLLMLogger.provider.info("← AFM Generable response generated successfully")
            SwiftLLMLogger.provider.debug("  Generated: \(String(describing: response.content), privacy: .public)")

            return response.content
        } catch {
            SwiftLLMLogger.error.logError(error, context: "AFM generateGenerable")
            throw LLMError.providerError(
                "AFM failed to generate Generable response: \(error.localizedDescription)",
                code: nil
            )
        }
    }

    public func streamCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) -> AsyncThrowingStream<String, Error> {
        let modelName = modelType == .onDevice ? "afm-on-device" : "afm-server"

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    SwiftLLMLogger.provider.info("→ AFM streamCompletion (\(modelName, privacy: .public))")
                    SwiftLLMLogger.provider.debug("  Prompt: \(prompt, privacy: .public)")

                    // Create session with system instructions
                    let session: LanguageModelSession
                    if let system = systemPrompt ?? instructions {
                        session = LanguageModelSession(instructions: system)
                    } else {
                        session = LanguageModelSession()
                    }

                    // Convert our GenerationOptions to AFM's GenerationOptions
                    let genOptions = FoundationModels.GenerationOptions(
                        sampling: nil,
                        temperature: options.temperature,
                        maximumResponseTokens: options.maxTokens
                    )

                    // Stream the response
                    let stream = session.streamResponse(options: genOptions) {
                        prompt
                    }

                    var chunkCount = 0
                    // Iterate over the stream and yield content
                    for try await snapshot in stream {
                        continuation.yield(snapshot.content)
                        chunkCount += 1
                    }

                    SwiftLLMLogger.provider.info("← AFM stream completed (\(chunkCount, privacy: .public) chunks)")
                    continuation.finish()
                } catch {
                    SwiftLLMLogger.error.logError(error, context: "AFM streamCompletion")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func estimateTokens(_ text: String) async throws -> Int {
        estimateTokensSync(text)
    }

    private func estimateTokensSync(_ text: String) -> Int {
        // AFM uses similar tokenization to GPT models (~4 chars per token)
        return max(1, text.count / 4)
    }
}

// MARK: - Convenience Initializers

@available(macOS 26.0, iOS 26.0, *)
extension AppleIntelligenceProvider {
    /// Create provider for on-device AFM (3B parameters)
    /// Fast, private, runs entirely on-device with no cost
    public static func onDevice(instructions: String? = nil) -> AppleIntelligenceProvider {
        AppleIntelligenceProvider(modelType: .onDevice, instructions: instructions)
    }

    /// Create provider for server-based AFM (larger mixture-of-experts)
    /// More powerful for complex tasks, requires network connection
    public static func server(instructions: String? = nil) -> AppleIntelligenceProvider {
        AppleIntelligenceProvider(modelType: .server, instructions: instructions)
    }
}

#else
// Implementation for when FoundationModels framework is not available
public struct AppleIntelligenceProvider: LLMProvider {
    public let id: String = "apple"
    public let displayName: String
    private let modelType: AFMModelType

    public enum AFMModelType: Sendable {
        case onDevice
        case server

        var displayName: String {
            switch self {
            case .onDevice: return "Apple Intelligence (On-Device)"
            case .server: return "Apple Intelligence (Server)"
            }
        }
    }

    public var capabilities: LLMCapabilities {
        LLMCapabilities(
            supportsStructuredOutput: false,
            supportsStreaming: false,
            supportsLocalExecution: false,
            supportsVision: false,
            supportsToolCalling: false,
            maxContextTokens: 0,
            maxOutputTokens: 0,
            supportsSystemPrompts: false,
            pricing: nil
        )
    }

    public init(modelType: AFMModelType = .onDevice, instructions: String? = nil) {
        self.modelType = modelType
        self.displayName = modelType.displayName
    }

    public func generateCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) async throws -> CompletionResponse {
        throw LLMError.unsupportedFeature("Apple Foundation Models require macOS 26+ or iOS 26+ with FoundationModels framework")
    }

    public func generateStructuredOutput<T: Codable>(
        prompt: String,
        systemPrompt: String?,
        schema: T.Type,
        options: GenerationOptions
    ) async throws -> T {
        throw LLMError.unsupportedFeature("Apple Foundation Models require macOS 26+ or iOS 26+ with FoundationModels framework")
    }

    public func streamCompletion(
        prompt: String,
        systemPrompt: String?,
        options: GenerationOptions
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: LLMError.unsupportedFeature("Apple Foundation Models require macOS 26+ or iOS 26+ with FoundationModels framework"))
        }
    }

    public func estimateTokens(_ text: String) async throws -> Int {
        throw LLMError.unsupportedFeature("Apple Foundation Models require macOS 26+ or iOS 26+ with FoundationModels framework")
    }
}

// MARK: - Convenience Initializers

extension AppleIntelligenceProvider {
    /// Create provider for on-device AFM (3B parameters)
    public static func onDevice(instructions: String? = nil) -> AppleIntelligenceProvider {
        AppleIntelligenceProvider(modelType: .onDevice, instructions: instructions)
    }

    /// Create provider for server-based AFM (larger mixture-of-experts)
    public static func server(instructions: String? = nil) -> AppleIntelligenceProvider {
        AppleIntelligenceProvider(modelType: .server, instructions: instructions)
    }
}
#endif
