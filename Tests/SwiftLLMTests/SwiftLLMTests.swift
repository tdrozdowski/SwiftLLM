import Testing
@testable import SwiftLLM

/// Comprehensive tests for SwiftLLM core functionality
/// Note: These tests don't make real API calls to avoid costs and rate limits

// MARK: - Model Tests

@Test func generationOptionsDefaults() {
    let options = GenerationOptions()

    #expect(options.temperature == nil)
    #expect(options.maxTokens == nil)
    #expect(options.topP == nil)
    #expect(options.model == nil)
}

@Test func generationOptionsCustom() {
    let options = GenerationOptions(
        model: "gpt-5-1-instant",
        temperature: 0.7,
        maxTokens: 1000,
        topP: 0.9
    )

    #expect(options.model == "gpt-5-1-instant")
    #expect(options.temperature == 0.7)
    #expect(options.maxTokens == 1000)
    #expect(options.topP == 0.9)
}

@Test func tokenUsage() {
    let usage = TokenUsage(inputTokens: 100, outputTokens: 50)

    #expect(usage.inputTokens == 100)
    #expect(usage.outputTokens == 50)
    #expect(usage.totalTokens == 150)
}

@Test func completionResponse() {
    let usage = TokenUsage(inputTokens: 10, outputTokens: 20)
    let response = CompletionResponse(
        text: "Test response",
        model: "test-model",
        usage: usage,
        finishReason: "stop",
        metadata: ["key": "value"]
    )

    #expect(response.text == "Test response")
    #expect(response.model == "test-model")
    #expect(response.usage.totalTokens == 30)
    #expect(response.finishReason == "stop")
    #expect(response.metadata?["key"] as? String == "value")
}

@Test func llmCapabilities() {
    let pricing = LLMPricing(inputCostPer1M: 5.0, outputCostPer1M: 15.0)
    let capabilities = LLMCapabilities(
        supportsStructuredOutput: true,
        supportsStreaming: true,
        supportsLocalExecution: false,
        supportsVision: true,
        supportsToolCalling: true,
        maxContextTokens: 200_000,
        maxOutputTokens: 8192,
        supportsSystemPrompts: true,
        pricing: pricing
    )

    #expect(capabilities.supportsStructuredOutput == true)
    #expect(capabilities.supportsStreaming == true)
    #expect(capabilities.supportsLocalExecution == false)
    #expect(capabilities.supportsVision == true)
    #expect(capabilities.supportsToolCalling == true)
    #expect(capabilities.maxContextTokens == 200_000)
    #expect(capabilities.maxOutputTokens == 8192)
    #expect(capabilities.supportsSystemPrompts == true)
    #expect(capabilities.pricing?.inputCostPer1M == 5.0)
    #expect(capabilities.pricing?.outputCostPer1M == 15.0)
}

@Test func llmPricing() {
    let pricing = LLMPricing(inputCostPer1M: 3.0, outputCostPer1M: 15.0)

    #expect(pricing.inputCostPer1M == 3.0)
    #expect(pricing.outputCostPer1M == 15.0)

    // Test cost calculation
    let inputTokens = 1000
    let outputTokens = 500
    let inputCost = Double(inputTokens) * pricing.inputCostPer1M / 1_000_000
    let outputCost = Double(outputTokens) * pricing.outputCostPer1M / 1_000_000
    let totalCost = inputCost + outputCost

    #expect(abs(inputCost - 0.003) < 0.0001)
    #expect(abs(outputCost - 0.0075) < 0.0001)
    #expect(abs(totalCost - 0.0105) < 0.0001)
}

// MARK: - Provider Capability Tests

@Test func anthropicProviderCapabilities() {
    let provider = AnthropicProvider(apiKey: "test-key", model: "claude-opus-4-5-20251124")

    #expect(provider.id == "anthropic")
    #expect(provider.capabilities.supportsStreaming == true)
    #expect(provider.capabilities.supportsVision == true) // Claude 4 supports vision
    #expect(provider.capabilities.supportsToolCalling == true)
    #expect(provider.capabilities.supportsLocalExecution == false)
    #expect(provider.capabilities.maxContextTokens == 200_000)
    #expect(provider.capabilities.maxOutputTokens == 8192)
    #expect(provider.capabilities.pricing != nil)
    #expect(provider.capabilities.pricing?.inputCostPer1M == 5.0)
    #expect(provider.capabilities.pricing?.outputCostPer1M == 25.0)
}

@Test func openAIProviderCapabilities() {
    let provider = OpenAIProvider(apiKey: "test-key", model: "gpt-5-1-instant")

    #expect(provider.id == "openai")
    #expect(provider.capabilities.supportsStructuredOutput == true)
    #expect(provider.capabilities.supportsStreaming == true)
    #expect(provider.capabilities.supportsVision == true)
    #expect(provider.capabilities.supportsToolCalling == true)
    #expect(provider.capabilities.supportsLocalExecution == false)
    #expect(provider.capabilities.maxContextTokens == 200_000)
    #expect(provider.capabilities.maxOutputTokens == 16_384)
    #expect(provider.capabilities.pricing != nil)
}

@Test func xaiProviderCapabilities() {
    let fastProvider = XAIProvider(apiKey: "test-key", model: "grok-4-1-fast-non-reasoning")

    #expect(fastProvider.id == "xai")
    #expect(fastProvider.capabilities.supportsStreaming == true)
    #expect(fastProvider.capabilities.supportsToolCalling == true)
    #expect(fastProvider.capabilities.supportsLocalExecution == false)
    #expect(fastProvider.capabilities.maxContextTokens == 2_000_000) // 2M context for Fast
    #expect(fastProvider.capabilities.pricing != nil)
    #expect(fastProvider.capabilities.pricing?.inputCostPer1M == 0.20)
    #expect(fastProvider.capabilities.pricing?.outputCostPer1M == 0.50)
}

@available(macOS 26.0, iOS 26.0, *)
@Test func appleIntelligenceProviderCapabilities() {
    let onDevice = AppleIntelligenceProvider(modelType: .onDevice)

    #expect(onDevice.id == "apple")
    #expect(onDevice.capabilities.supportsStreaming == true)
    #expect(onDevice.capabilities.supportsStructuredOutput == true)
    #expect(onDevice.capabilities.supportsVision == true)
    #expect(onDevice.capabilities.supportsToolCalling == true)
    #expect(onDevice.capabilities.supportsLocalExecution == true)
    #expect(onDevice.capabilities.maxContextTokens == 8192)
    #expect(onDevice.capabilities.maxOutputTokens == 4096)
    #expect(onDevice.capabilities.pricing == nil) // Free on-device

    let server = AppleIntelligenceProvider(modelType: .server)
    #expect(server.capabilities.supportsLocalExecution == false)
    #expect(server.capabilities.maxContextTokens == 32_768)
}

// MARK: - Convenience Initializer Tests

@Test func anthropicConvenienceInitializers() {
    let opus45 = AnthropicProvider.opus45(apiKey: "test")
    #expect(opus45.displayName == "Anthropic Claude")

    let sonnet45 = AnthropicProvider.sonnet45(apiKey: "test")
    #expect(sonnet45.displayName == "Anthropic Claude")

    let haiku45 = AnthropicProvider.haiku45(apiKey: "test")
    #expect(haiku45.displayName == "Anthropic Claude")
}

@Test func openAIConvenienceInitializers() {
    let instant = OpenAIProvider.gpt51Instant(apiKey: "test")
    #expect(instant.displayName == "OpenAI GPT")

    let thinking = OpenAIProvider.gpt51Thinking(apiKey: "test")
    #expect(thinking.displayName == "OpenAI GPT")

    let codexMax = OpenAIProvider.gpt51CodexMax(apiKey: "test")
    #expect(codexMax.displayName == "OpenAI GPT")
}

@Test func xaiConvenienceInitializers() {
    let thinking = XAIProvider.grok41Thinking(apiKey: "test")
    #expect(thinking.displayName == "xAI Grok")

    let fast = XAIProvider.grok41FastNonReasoning(apiKey: "test")
    #expect(fast.displayName == "xAI Grok")
}

@available(macOS 26.0, iOS 26.0, *)
@Test func appleConvenienceInitializers() {
    let onDevice = AppleIntelligenceProvider.onDevice()
    #expect(onDevice.displayName == "Apple Intelligence (On-Device)")

    let server = AppleIntelligenceProvider.server()
    #expect(server.displayName == "Apple Intelligence (Server)")
}

// MARK: - Token Estimation Tests

@available(macOS 26.0, iOS 26.0, *)
@Test func tokenEstimation() async throws {
    let provider = AppleIntelligenceProvider.onDevice()

    let shortText = "Hello"
    let shortEstimate = try await provider.estimateTokens(shortText)
    #expect(shortEstimate > 0)
    #expect(shortEstimate < 10)

    let longText = String(repeating: "test ", count: 100)
    let longEstimate = try await provider.estimateTokens(longText)
    #expect(longEstimate > shortEstimate)
}

// MARK: - Context Window Tests

@Test func contextWindowSizes() {
    // Test different model context windows
    let claudeOpus = AnthropicProvider.opus45(apiKey: "test")
    #expect(claudeOpus.capabilities.maxContextTokens == 200_000)

    let gpt51Instant = OpenAIProvider.gpt51Instant(apiKey: "test")
    #expect(gpt51Instant.capabilities.maxContextTokens == 200_000)

    let gpt51CodexMax = OpenAIProvider.gpt51CodexMax(apiKey: "test")
    #expect(gpt51CodexMax.capabilities.maxContextTokens == 1_000_000)

    let grokFast = XAIProvider.grok41FastNonReasoning(apiKey: "test")
    #expect(grokFast.capabilities.maxContextTokens == 2_000_000)

    if #available(macOS 26.0, iOS 26.0, *) {
        let appleOnDevice = AppleIntelligenceProvider.onDevice()
        #expect(appleOnDevice.capabilities.maxContextTokens == 8192)

        let appleServer = AppleIntelligenceProvider.server()
        #expect(appleServer.capabilities.maxContextTokens == 32_768)
    }
}

// MARK: - Pricing Tests

@Test func providerPricing() {
    // Anthropic pricing
    let claudeOpus45 = AnthropicProvider.opus45(apiKey: "test")
    #expect(claudeOpus45.capabilities.pricing?.inputCostPer1M == 5.0)
    #expect(claudeOpus45.capabilities.pricing?.outputCostPer1M == 25.0)

    let claudeHaiku45 = AnthropicProvider.haiku45(apiKey: "test")
    #expect(claudeHaiku45.capabilities.pricing?.inputCostPer1M == 1.0)
    #expect(claudeHaiku45.capabilities.pricing?.outputCostPer1M == 5.0)

    // OpenAI pricing
    let gpt51 = OpenAIProvider.gpt51Instant(apiKey: "test")
    #expect(gpt51.capabilities.pricing?.inputCostPer1M == 5.0)
    #expect(gpt51.capabilities.pricing?.outputCostPer1M == 15.0)

    // xAI pricing
    let grokFast = XAIProvider.grok41FastNonReasoning(apiKey: "test")
    #expect(grokFast.capabilities.pricing?.inputCostPer1M == 0.20)
    #expect(grokFast.capabilities.pricing?.outputCostPer1M == 0.50)

    // Apple pricing (free on-device)
    if #available(macOS 26.0, iOS 26.0, *) {
        let apple = AppleIntelligenceProvider.onDevice()
        #expect(apple.capabilities.pricing == nil)
    }
}

// MARK: - Protocol Conformance Tests

@Test func providerProtocolConformance() {
    var providers: [any LLMProvider] = [
        AnthropicProvider.opus45(apiKey: "test"),
        OpenAIProvider.gpt51Instant(apiKey: "test"),
        XAIProvider.grok41Thinking(apiKey: "test")
    ]

    if #available(macOS 26.0, iOS 26.0, *) {
        providers.append(AppleIntelligenceProvider.onDevice())
    }

    for provider in providers {
        #expect(!provider.id.isEmpty)
        #expect(!provider.displayName.isEmpty)
        #expect(provider.capabilities.maxContextTokens > 0)
        #expect(provider.capabilities.maxOutputTokens > 0)
    }
}

// MARK: - Sendable Conformance Tests

@available(macOS 26.0, iOS 26.0, *)
@Test func sendableConformance() async {
    // Test that providers can be safely used across concurrency boundaries
    let provider = AppleIntelligenceProvider.onDevice()

    await withTaskGroup(of: Void.self) { group in
        for _ in 0..<10 {
            group.addTask {
                let caps = provider.capabilities
                #expect(caps.maxContextTokens > 0)
            }
        }
    }
}

// MARK: - Model Name Mapping Tests

@Test func anthropicModelMapping() {
    let opus4 = AnthropicProvider.opus4(apiKey: "test")
    #expect(opus4.capabilities.maxContextTokens == 200_000)

    let sonnet4 = AnthropicProvider.sonnet4(apiKey: "test")
    #expect(sonnet4.capabilities.maxContextTokens == 200_000)
}

@Test func openAIModelMapping() {
    let gpt4o = OpenAIProvider.gpt4o(apiKey: "test")
    #expect(gpt4o.capabilities.maxContextTokens == 128_000)

    let gpt5 = OpenAIProvider.gpt5(apiKey: "test")
    #expect(gpt5.capabilities.maxContextTokens == 200_000)
}

@Test func xaiModelMapping() {
    let grok2 = XAIProvider.grok2(apiKey: "test")
    #expect(grok2.capabilities.maxContextTokens == 131_072)

    let grok41 = XAIProvider.grok41(apiKey: "test")
    #expect(grok41.capabilities.maxContextTokens == 500_000)
}
