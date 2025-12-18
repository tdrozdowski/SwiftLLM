import Foundation
import SwiftLLM

// Manual test for xAI - run this to see logs
// Usage: swift run or create an executable target

@main
struct XAIManualTest {
    static func main() async {
        print("🧪 Testing xAI integration...")
        print("📋 Check logs with: log stream --predicate 'subsystem == \"com.swiftllm\"' --level debug")
        print("")

        // Load API key from environment
        guard let apiKey = ProcessInfo.processInfo.environment["XAI_API_KEY"] else {
            print("❌ XAI_API_KEY not found in environment")
            print("   Run: export XAI_API_KEY=your-key-here")
            return
        }

        print("✅ API key loaded: \(String(apiKey.prefix(10)))...")

        do {
            // Test 1: Simple completion
            print("\n--- Test 1: Simple Completion ---")
            let provider = XAIProvider(apiKey: apiKey, model: "grok-4-1-fast-non-reasoning")

            let response = try await provider.generateCompletion(
                prompt: "Say hello in one word",
                systemPrompt: nil as String?,
                options: GenerationOptions.default
            )

            print("✅ Response: \(response.text)")
            print("   Model: \(response.model)")
            print("   Tokens: \(response.usage.totalTokens)")

            // Test 2: Structured output (this is where markdown wrapping would show up)
            print("\n--- Test 2: Structured Output (JSON) ---")

            struct Greeting: Codable {
                let greeting: String
            }

            let structured = try await provider.generateStructuredOutput(
                prompt: "Return JSON with a 'greeting' field containing a hello message",
                systemPrompt: nil as String?,
                schema: Greeting.self,
                options: GenerationOptions.default
            )

            print("✅ Structured output:")
            print("   Greeting: \(structured.greeting)")

            print("\n🎉 All tests passed!")

        } catch {
            print("\n❌ Test failed: \(error)")
        }
    }
}
