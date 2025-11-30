import SwiftLLM
import Foundation

/// Basic usage examples for all SwiftLLM providers
/// These examples demonstrate the fundamental operations: completions, streaming, and structured output

// MARK: - Anthropic Claude Examples

func anthropicBasicExample() async throws {
    print("=== Anthropic Claude Opus 4.5 Example ===\n")

    let provider = AnthropicProvider.opus45(apiKey: "your-anthropic-api-key")

    let response = try await provider.generateCompletion(
        prompt: "Explain the concept of closures in Swift programming",
        systemPrompt: "You are an expert Swift developer and educator",
        options: GenerationOptions(
            temperature: 0.7,
            maxTokens: 500
        )
    )

    print("Response: \(response.text)")
    print("\nUsage:")
    print("  Input tokens: \(response.usage.inputTokens)")
    print("  Output tokens: \(response.usage.outputTokens)")
    print("  Total tokens: \(response.usage.totalTokens)")
}

func anthropicStreamingExample() async throws {
    print("\n=== Anthropic Claude Sonnet 4.5 Streaming ===\n")

    let provider = AnthropicProvider.sonnet45(apiKey: "your-anthropic-api-key")

    print("Streaming response: ")

    let stream = provider.streamCompletion(
        prompt: "Write a haiku about programming",
        systemPrompt: "You are a creative poet",
        options: GenerationOptions(temperature: 0.9)
    )

    for try await chunk in stream {
        print(chunk, terminator: "")
    }
    print("\n")
}

// MARK: - OpenAI GPT Examples

func openAIBasicExample() async throws {
    print("=== OpenAI GPT-5.1 Instant Example ===\n")

    let provider = OpenAIProvider.gpt51Instant(apiKey: "your-openai-api-key")

    let response = try await provider.generateCompletion(
        prompt: "Write a function to calculate fibonacci numbers",
        systemPrompt: "You are an expert programmer. Provide clean, well-documented code.",
        options: GenerationOptions(
            temperature: 0.3,
            maxTokens: 1000
        )
    )

    print("Response: \(response.text)")
    print("\nModel: \(response.model)")
    print("Finish reason: \(response.finishReason ?? "none")")
}

func openAIStructuredOutputExample() async throws {
    print("\n=== OpenAI GPT-5.1 Structured Output ===\n")

    struct Book: Codable {
        let title: String
        let author: String
        let genre: String
        let publicationYear: Int
        let summary: String
    }

    let provider = OpenAIProvider.gpt51Instant(apiKey: "your-openai-api-key")

    let book = try await provider.generateStructuredOutput(
        prompt: "Generate information about a classic science fiction novel",
        systemPrompt: "You are a literary expert",
        schema: Book.self,
        options: GenerationOptions(temperature: 0.7)
    )

    print("Title: \(book.title)")
    print("Author: \(book.author)")
    print("Genre: \(book.genre)")
    print("Year: \(book.publicationYear)")
    print("Summary: \(book.summary)")
}

// MARK: - xAI Grok Examples

func xAIBasicExample() async throws {
    print("\n=== xAI Grok 4.1 Thinking Example ===\n")

    let provider = XAIProvider.grok41Thinking(apiKey: "your-xai-api-key")

    let response = try await provider.generateCompletion(
        prompt: "Explain the potential impact of quantum computing on current encryption methods",
        systemPrompt: "You are a cybersecurity expert with deep knowledge of quantum computing",
        options: GenerationOptions(
            temperature: 0.5,
            maxTokens: 800
        )
    )

    print("Response: \(response.text)")
    print("\nUsage:")
    print("  Input tokens: \(response.usage.inputTokens)")
    print("  Output tokens: \(response.usage.outputTokens)")

    // Calculate cost (Grok 4.1 Thinking pricing)
    if let pricing = provider.capabilities.pricing {
        let inputCost = Double(response.usage.inputTokens) * pricing.inputCostPer1M / 1_000_000
        let outputCost = Double(response.usage.outputTokens) * pricing.outputCostPer1M / 1_000_000
        let totalCost = inputCost + outputCost
        print("  Estimated cost: $\(String(format: "%.6f", totalCost))")
    }
}

func xAIFastExample() async throws {
    print("\n=== xAI Grok 4.1 Fast (2M Context) ===\n")

    let provider = XAIProvider.grok41FastNonReasoning(apiKey: "your-xai-api-key")

    print("Provider capabilities:")
    print("  Max context: \(provider.capabilities.maxContextTokens) tokens")
    print("  Supports streaming: \(provider.capabilities.supportsStreaming)")
    print("  Supports tool calling: \(provider.capabilities.supportsToolCalling)")

    let response = try await provider.generateCompletion(
        prompt: "Summarize the key features of SwiftUI",
        systemPrompt: nil,
        options: GenerationOptions(temperature: 0.3)
    )

    print("\nResponse: \(response.text)")
}

// MARK: - Local LLM Examples (Ollama, LM Studio, etc.)

func ollamaBasicExample() async throws {
    print("\n=== Ollama Local LLM Example ===\n")

    // Simple usage with just model name
    let provider = LocalLLMProvider.ollama(model: "llama3.2")

    print("Provider capabilities:")
    print("  Local execution: \(provider.capabilities.supportsLocalExecution)")
    print("  Cost: Free (runs locally)")

    let response = try await provider.generateCompletion(
        prompt: "Explain what makes Rust memory-safe",
        systemPrompt: "You are a systems programming expert",
        options: GenerationOptions(
            temperature: 0.7,
            maxTokens: 500
        )
    )

    print("\nResponse: \(response.text)")
    print("Model: \(response.model)")
}

func ollamaCustomModelExample() async throws {
    print("\n=== Ollama Custom Model Configuration ===\n")

    // Full control over model capabilities
    let customModel = LocalModelConfig(
        name: "deepseek-coder-v2:16b",
        contextWindow: 128_000,
        supportsVision: false,
        supportsToolCalling: false,
        supportsStructuredOutput: true
    )

    let provider = LocalLLMProvider.ollama(model: customModel)

    print("Custom model config:")
    print("  Name: \(customModel.name)")
    print("  Context window: \(customModel.contextWindow)")
    print("  Max context: \(provider.capabilities.maxContextTokens) tokens")

    let response = try await provider.generateCompletion(
        prompt: "Write a Swift function to parse JSON with error handling",
        systemPrompt: "You are an expert Swift developer",
        options: GenerationOptions(temperature: 0.3)
    )

    print("\nResponse: \(response.text)")
}

func lmStudioExample() async throws {
    print("\n=== LM Studio Example ===\n")

    // LM Studio runs on port 1234 by default
    let provider = LocalLLMProvider.lmStudio(model: "mistral-7b-instruct")

    let stream = provider.streamCompletion(
        prompt: "Write a poem about coding at midnight",
        systemPrompt: "You are a creative writer",
        options: GenerationOptions(temperature: 0.9)
    )

    print("Streaming response: ")
    for try await chunk in stream {
        print(chunk, terminator: "")
    }
    print("\n")
}

func openAICompatibleServerExample() async throws {
    print("\n=== Generic OpenAI-Compatible Server ===\n")

    // Works with any server that implements the OpenAI API
    // Examples: vLLM, text-generation-inference, FastChat, etc.
    let provider = LocalLLMProvider.openAICompatible(
        baseURL: "http://my-gpu-server:8000",
        model: LocalModelConfig(
            name: "my-fine-tuned-llama",
            contextWindow: 32768,
            supportsVision: false,
            supportsToolCalling: true,
            supportsStructuredOutput: true
        ),
        apiKey: "optional-api-key"
    )

    let response = try await provider.generateCompletion(
        prompt: "Analyze this code for potential bugs",
        systemPrompt: nil,
        options: GenerationOptions()
    )

    print("Response: \(response.text)")
}

func localLLMStructuredOutputExample() async throws {
    print("\n=== Local LLM Structured Output ===\n")

    struct CodeReview: Codable {
        let summary: String
        let issues: [String]
        let suggestions: [String]
        let rating: Int
    }

    let provider = LocalLLMProvider.ollama(
        model: LocalModelConfig(
            name: "llama3.2",
            contextWindow: 128_000,
            supportsStructuredOutput: true
        )
    )

    let review = try await provider.generateStructuredOutput(
        prompt: """
        Review this code:
        func add(a: Int, b: Int) -> Int {
            return a + b
        }
        """,
        systemPrompt: "You are a code reviewer. Respond with JSON.",
        schema: CodeReview.self,
        options: GenerationOptions(temperature: 0.3)
    )

    print("Summary: \(review.summary)")
    print("Issues: \(review.issues)")
    print("Suggestions: \(review.suggestions)")
    print("Rating: \(review.rating)/10")
}

// MARK: - Apple Foundation Models Examples

@available(macOS 26.0, iOS 26.0, *)
func appleOnDeviceExample() async throws {
    print("\n=== Apple Foundation Models (On-Device) ===\n")

    let provider = AppleIntelligenceProvider.onDevice(
        instructions: "You are a helpful writing assistant"
    )

    print("Provider capabilities:")
    print("  Local execution: \(provider.capabilities.supportsLocalExecution)")
    print("  Privacy: Complete (never leaves device)")
    print("  Cost: Free")
    print("  Max context: \(provider.capabilities.maxContextTokens) tokens")

    let response = try await provider.generateCompletion(
        prompt: "Draft a professional email requesting a meeting with a potential client",
        systemPrompt: nil,
        options: GenerationOptions(temperature: 0.7)
    )

    print("\nResponse: \(response.text)")
    print("\nThis ran entirely on your device - no data sent to servers!")
}

@available(macOS 26.0, iOS 26.0, *)
func appleServerExample() async throws {
    print("\n=== Apple Foundation Models (Server) ===\n")

    let provider = AppleIntelligenceProvider.server(
        instructions: "You are an expert software architect"
    )

    print("Provider capabilities:")
    print("  Max context: \(provider.capabilities.maxContextTokens) tokens")
    print("  Supports vision: \(provider.capabilities.supportsVision)")
    print("  Supports structured output: \(provider.capabilities.supportsStructuredOutput)")

    let response = try await provider.generateCompletion(
        prompt: "Design a scalable microservices architecture for an e-commerce platform",
        systemPrompt: nil,
        options: GenerationOptions(
            temperature: 0.5,
            maxTokens: 2000
        )
    )

    print("\nResponse: \(response.text)")
}

// MARK: - Error Handling Example

func errorHandlingExample() async {
    print("\n=== Error Handling Example ===\n")

    let provider = AnthropicProvider.opus45(apiKey: "invalid-key")

    do {
        let _ = try await provider.generateCompletion(
            prompt: "Hello",
            systemPrompt: nil,
            options: GenerationOptions()
        )
    } catch LLMError.authenticationFailed(let message) {
        print("❌ Authentication failed: \(message)")
    } catch LLMError.rateLimitExceeded(let retryAfter) {
        if let retryTime = retryAfter {
            print("⏳ Rate limited. Retry after \(retryTime) seconds")
        } else {
            print("⏳ Rate limited. Please try again later")
        }
    } catch LLMError.invalidRequest(let message) {
        print("⚠️ Invalid request: \(message)")
    } catch LLMError.networkError(let error) {
        print("🌐 Network error: \(error.localizedDescription)")
    } catch LLMError.providerError(let message, let code) {
        print("🔥 Provider error [\(code ?? "unknown")]: \(message)")
    } catch LLMError.decodingError(let message) {
        print("📦 Decoding error: \(message)")
    } catch LLMError.unsupportedFeature(let message) {
        print("⛔ Unsupported feature: \(message)")
    } catch {
        print("❓ Unexpected error: \(error)")
    }
}

// MARK: - Provider Comparison Example

func providerComparisonExample() async throws {
    print("\n=== Provider Comparison ===\n")

    let prompt = "What is the capital of France?"
    let systemPrompt = "You are a geography expert"
    let options = GenerationOptions(temperature: 0.3, maxTokens: 50)

    // Anthropic
    print("Testing Anthropic Claude Haiku 4.5 (fast & cheap)...")
    let claude = AnthropicProvider.haiku45(apiKey: "your-key")
    let claudeStart = Date()
    let claudeResponse = try await claude.generateCompletion(
        prompt: prompt,
        systemPrompt: systemPrompt,
        options: options
    )
    let claudeDuration = Date().timeIntervalSince(claudeStart)

    // OpenAI
    print("Testing OpenAI GPT-5.1 Instant...")
    let openai = OpenAIProvider.gpt51Instant(apiKey: "your-key")
    let openaiStart = Date()
    let openaiResponse = try await openai.generateCompletion(
        prompt: prompt,
        systemPrompt: systemPrompt,
        options: options
    )
    let openaiDuration = Date().timeIntervalSince(openaiStart)

    // xAI
    print("Testing xAI Grok 4.1 Fast...")
    let grok = XAIProvider.grok41FastNonReasoning(apiKey: "your-key")
    let grokStart = Date()
    let grokResponse = try await grok.generateCompletion(
        prompt: prompt,
        systemPrompt: systemPrompt,
        options: options
    )
    let grokDuration = Date().timeIntervalSince(grokStart)

    // Results
    print("\n📊 Comparison Results:\n")
    print("Claude Haiku 4.5:")
    print("  Time: \(String(format: "%.2f", claudeDuration))s")
    print("  Tokens: \(claudeResponse.usage.totalTokens)")
    print("  Response: \(claudeResponse.text)")

    print("\nGPT-5.1 Instant:")
    print("  Time: \(String(format: "%.2f", openaiDuration))s")
    print("  Tokens: \(openaiResponse.usage.totalTokens)")
    print("  Response: \(openaiResponse.text)")

    print("\nGrok 4.1 Fast:")
    print("  Time: \(String(format: "%.2f", grokDuration))s")
    print("  Tokens: \(grokResponse.usage.totalTokens)")
    print("  Response: \(grokResponse.text)")
}

// MARK: - Main Function

@main
struct BasicUsageExamples {
    static func main() async {
        print("SwiftLLM Basic Usage Examples\n")
        print("=" * 50 + "\n")

        do {
            // Uncomment the examples you want to run

            // try await anthropicBasicExample()
            // try await anthropicStreamingExample()

            // try await openAIBasicExample()
            // try await openAIStructuredOutputExample()

            // try await xAIBasicExample()
            // try await xAIFastExample()

            // Local LLMs (Ollama, LM Studio, etc.):
            // try await ollamaBasicExample()
            // try await ollamaCustomModelExample()
            // try await lmStudioExample()
            // try await openAICompatibleServerExample()
            // try await localLLMStructuredOutputExample()

            // On macOS 26+ / iOS 26+:
            // if #available(macOS 26.0, iOS 26.0, *) {
            //     try await appleOnDeviceExample()
            //     try await appleServerExample()
            // }

            // await errorHandlingExample()

            // try await providerComparisonExample()

            print("\n✅ Examples completed successfully!")

        } catch {
            print("\n❌ Error running examples: \(error)")
        }
    }
}
