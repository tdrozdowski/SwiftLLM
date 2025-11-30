import XCTest
@testable import SwiftLLM

/// Performance benchmarks for SwiftLLM
/// Run with: swift test --filter SwiftLLMBenchmarks
final class SwiftLLMBenchmarks: XCTestCase {

    // MARK: - Token Estimation Benchmarks

    func testTokenEstimationPerformance_Short() throws {
        let provider = OpenAIProvider(apiKey: "test", model: "gpt-4o")
        let shortText = "Hello, world!"

        measure {
            for _ in 0..<1000 {
                _ = Task {
                    _ = try? await provider.estimateTokens(shortText)
                }
            }
        }
    }

    func testTokenEstimationPerformance_Long() throws {
        let provider = OpenAIProvider(apiKey: "test", model: "gpt-4o")
        let longText = String(repeating: "This is a test sentence for token estimation. ", count: 1000)

        measure {
            for _ in 0..<100 {
                _ = Task {
                    _ = try? await provider.estimateTokens(longText)
                }
            }
        }
    }

    // MARK: - Usage Tracker Benchmarks

    func testUsageTrackerRecordingPerformance() async throws {
        let tracker = UsageTracker()

        let usage = TokenUsage(inputTokens: 100, outputTokens: 50)
        let record = UsageTracker.UsageRecord(
            providerId: "test",
            model: "test-model",
            usage: usage,
            cost: 0.001
        )

        let startTime = CFAbsoluteTimeGetCurrent()

        for _ in 0..<10000 {
            await tracker.record(record)
        }

        let duration = CFAbsoluteTimeGetCurrent() - startTime
        print("Recording 10,000 records took: \(String(format: "%.4f", duration))s")

        // Should complete in under 1 second
        XCTAssertLessThan(duration, 1.0, "Recording should be fast")
    }

    func testUsageTrackerStatsPerformance() async throws {
        let tracker = UsageTracker()

        // Pre-populate with records
        for i in 0..<10000 {
            let usage = TokenUsage(inputTokens: 100 + i % 100, outputTokens: 50 + i % 50)
            let record = UsageTracker.UsageRecord(
                providerId: ["anthropic", "openai", "xai"][i % 3],
                model: ["claude", "gpt", "grok"][i % 3],
                usage: usage,
                cost: Double(i) * 0.0001
            )
            await tracker.record(record)
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        for _ in 0..<100 {
            _ = await tracker.stats()
        }

        let duration = CFAbsoluteTimeGetCurrent() - startTime
        print("Computing stats 100 times over 10,000 records took: \(String(format: "%.4f", duration))s")

        // Should complete in under 2 seconds
        XCTAssertLessThan(duration, 2.0, "Stats computation should be reasonably fast")
    }

    // MARK: - Provider Registry Benchmarks

    func testProviderRegistryLookupPerformance() throws {
        let registry = ProviderRegistry()

        // Register multiple providers
        for i in 0..<100 {
            registry.register(
                ProviderRegistry.ProviderConfig(type: .ollama, model: "model-\(i)"),
                as: "provider-\(i)"
            )
        }

        measure {
            for i in 0..<10000 {
                _ = registry.provider(named: "provider-\(i % 100)")
            }
        }
    }

    // MARK: - Model Creation Benchmarks

    func testGenerationOptionsCreation() throws {
        measure {
            for _ in 0..<10000 {
                _ = GenerationOptions(
                    temperature: 0.7,
                    maxTokens: 1000,
                    topP: 0.9,
                    frequencyPenalty: 0.5,
                    presencePenalty: 0.5,
                    stopSequences: ["stop1", "stop2"]
                )
            }
        }
    }

    func testCompletionResponseCreation() throws {
        let usage = TokenUsage(inputTokens: 100, outputTokens: 50)

        measure {
            for _ in 0..<10000 {
                _ = CompletionResponse(
                    text: "This is a test response",
                    model: "test-model",
                    usage: usage,
                    finishReason: "stop",
                    metadata: ["key": "value"]
                )
            }
        }
    }

    // MARK: - Cost Calculation Benchmarks

    func testCostCalculationPerformance() throws {
        let pricing = LLMPricing(inputCostPer1M: 15.0, outputCostPer1M: 75.0)
        let usage = TokenUsage(inputTokens: 1000, outputTokens: 500)

        measure {
            for _ in 0..<100000 {
                _ = usage.cost(with: pricing)
            }
        }
    }

    // MARK: - LocalModelConfig Benchmarks

    func testLocalModelConfigCreation() throws {
        measure {
            for _ in 0..<10000 {
                _ = LocalModelConfig(
                    name: "llama3.2",
                    contextWindow: 128_000,
                    supportsVision: false,
                    supportsToolCalling: true,
                    supportsStructuredOutput: true
                )
            }
        }
    }

    // MARK: - JSON Export/Import Benchmarks

    func testUsageTrackerJSONExport() async throws {
        let tracker = UsageTracker()

        // Pre-populate
        for i in 0..<1000 {
            let usage = TokenUsage(inputTokens: 100 + i, outputTokens: 50 + i)
            let record = UsageTracker.UsageRecord(
                providerId: "test",
                model: "test-model",
                usage: usage,
                cost: Double(i) * 0.001
            )
            await tracker.record(record)
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        for _ in 0..<10 {
            _ = try await tracker.exportJSON()
        }

        let duration = CFAbsoluteTimeGetCurrent() - startTime
        print("Exporting 1,000 records to JSON 10 times took: \(String(format: "%.4f", duration))s")

        XCTAssertLessThan(duration, 2.0, "JSON export should be fast")
    }
}
