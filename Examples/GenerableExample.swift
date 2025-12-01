import Foundation
import SwiftLLM

#if canImport(FoundationModels)
import FoundationModels

@available(macOS 26.0, iOS 26.0, *)
struct GenerableExamples {

    // MARK: - Example 1: Simple Code Summary

    @Generable
    struct CodeSummary {
        @Guide(description: "One-line summary of what the code does")
        var brief: String

        @Guide(description: "Key points about the implementation")
        var keyPoints: [String]

        @Guide(description: "Potential improvements or concerns")
        var suggestions: [String]
    }

    static func analyzeCode() async throws {
        let provider = AppleIntelligenceProvider.onDevice(
            instructions: "You are an expert code reviewer"
        )

        let code = """
        func fibonacci(_ n: Int) -> Int {
            guard n > 1 else { return n }
            return fibonacci(n - 1) + fibonacci(n - 2)
        }
        """

        let summary = try await provider.generateGenerable(
            prompt: "Analyze this Swift code:\n\(code)",
            systemPrompt: nil,
            responseType: CodeSummary.self,
            options: .default
        )

        print("Summary: \(summary.brief)")
        print("Key Points:")
        for point in summary.keyPoints {
            print("  - \(point)")
        }
        print("Suggestions:")
        for suggestion in summary.suggestions {
            print("  - \(suggestion)")
        }
    }

    // MARK: - Example 2: Structured Data Extraction

    @Generable
    struct EmailAnalysis {
        @Guide(description: "The sender's email address")
        var sender: String

        @Guide(description: "List of recipient email addresses")
        var recipients: [String]

        @Guide(description: "Email subject line")
        var subject: String

        @Guide(description: "Action items mentioned in the email")
        var actionItems: [String]

        @Guide(description: "Priority level: low, medium, or high")
        var priority: String
    }

    static func analyzeEmail() async throws {
        let provider = AppleIntelligenceProvider.server()

        let emailText = """
        From: alice@example.com
        To: bob@example.com, charlie@example.com
        Subject: Q4 Planning Meeting - Action Items

        Hi team,

        Following up on today's meeting, here are the action items:
        1. Bob will prepare the budget proposal by Friday
        2. Charlie will schedule interviews for the new hire
        3. I'll send the client presentation draft by Wednesday

        This is high priority - we need everything ready for next week's board meeting.

        Best,
        Alice
        """

        let analysis = try await provider.generateGenerable(
            prompt: "Extract structured information from this email:\n\(emailText)",
            systemPrompt: "You are an email analysis assistant",
            responseType: EmailAnalysis.self,
            options: .default
        )

        print("From: \(analysis.sender)")
        print("To: \(analysis.recipients.joined(separator: ", "))")
        print("Subject: \(analysis.subject)")
        print("Priority: \(analysis.priority)")
        print("\nAction Items:")
        for (index, item) in analysis.actionItems.enumerated() {
            print("  \(index + 1). \(item)")
        }
    }

    // MARK: - Example 3: Nested Structures

    @Generable
    struct ProductReview {
        @Guide(description: "Overall rating from 1-5")
        var rating: Int

        @Guide(description: "Detailed breakdown by category")
        var breakdown: ReviewBreakdown

        @Guide(description: "Recommended for purchase?")
        var recommended: Bool
    }

    @Generable
    struct ReviewBreakdown {
        @Guide(description: "Quality rating 1-5")
        var quality: Int

        @Guide(description: "Value for money rating 1-5")
        var value: Int

        @Guide(description: "Ease of use rating 1-5")
        var usability: Int
    }

    static func analyzeProductReview() async throws {
        let provider = AppleIntelligenceProvider.onDevice()

        let reviewText = """
        I've been using this laptop for 3 months now. The build quality is excellent -
        solid aluminum construction. However, it's quite expensive for the specs you get.
        The keyboard takes some getting used to, but once you adjust, typing is comfortable.
        Overall, it's a good machine but maybe wait for a sale.
        """

        let review = try await provider.generateGenerable(
            prompt: "Analyze this product review and provide a structured assessment:\n\(reviewText)",
            systemPrompt: "You are a product review analyzer",
            responseType: ProductReview.self,
            options: GenerationOptions(
                temperature: 0.3,  // Lower temperature for more consistent ratings
                maxTokens: 500
            )
        )

        print("Overall Rating: \(review.rating)/5")
        print("Quality: \(review.breakdown.quality)/5")
        print("Value: \(review.breakdown.value)/5")
        print("Usability: \(review.breakdown.usability)/5")
        print("Recommended: \(review.recommended ? "Yes" : "No")")
    }

    // MARK: - Example 4: Comparison with JSON-based approach

    /// This demonstrates why @Generable is better than the JSON prompt hack
    static func compareApproaches() async throws {
        let provider = AppleIntelligenceProvider.onDevice()

        let prompt = "What are three benefits of Swift over Objective-C?"

        // OLD WAY: Using generateStructuredOutput (JSON prompt hack)
        // This is fragile - AFM might wrap JSON in markdown, add explanatory text, etc.
        struct Benefits: Codable {
            let benefits: [String]
        }

        let jsonResult = try await provider.generateStructuredOutput(
            prompt: prompt,
            systemPrompt: nil,
            schema: Benefits.self,
            options: .default
        )
        print("JSON approach result: \(jsonResult.benefits)")

        // NEW WAY: Using generateGenerable with @Generable
        // AFM's guided generation ensures the response conforms to the schema
        @Generable
        struct BenefitsList {
            @Guide(description: "List of three key benefits")
            var benefits: [String]
        }

        let generableResult = try await provider.generateGenerable(
            prompt: prompt,
            systemPrompt: nil,
            responseType: BenefitsList.self,
            options: .default
        )
        print("@Generable approach result: \(generableResult.benefits)")

        // The @Generable approach is:
        // 1. More reliable - no JSON parsing errors
        // 2. Type-safe at compile time
        // 3. Uses AFM's native capabilities
        // 4. No markdown wrapping issues
    }

    // MARK: - Example 5: Using with Provider Protocol

    static func providerProtocolExample() async throws {
        // When you know you're using AFM, call directly
        let afmProvider = AppleIntelligenceProvider.onDevice()

        @Generable
        struct Summary {
            @Guide(description: "Brief summary")
            var text: String
        }

        let result = try await afmProvider.generateGenerable(
            prompt: "Summarize: Swift is a modern programming language",
            systemPrompt: nil,
            responseType: Summary.self,
            options: .default
        )
        print(result.text)

        // If you have a protocol-typed provider, generateGenerable is available
        // but will throw for non-AFM providers
        let provider: any LLMProvider = afmProvider

        do {
            let protocolResult = try await provider.generateGenerable(
                prompt: "Summarize: Swift is awesome",
                systemPrompt: nil,
                responseType: Summary.self,
                options: .default
            )
            print(protocolResult.text)
        } catch {
            print("Error: \(error)")
            // For non-AFM providers, use generateStructuredOutput instead
        }
    }
}

// MARK: - Running Examples

@available(macOS 26.0, iOS 26.0, *)
@main
struct GenerableExamplesRunner {
    static func main() async {
        print("🧪 SwiftLLM @Generable Examples\n")

        do {
            print("--- Example 1: Code Analysis ---")
            try await GenerableExamples.analyzeCode()

            print("\n--- Example 2: Email Analysis ---")
            try await GenerableExamples.analyzeEmail()

            print("\n--- Example 3: Product Review ---")
            try await GenerableExamples.analyzeProductReview()

            print("\n--- Example 4: Approach Comparison ---")
            try await GenerableExamples.compareApproaches()

            print("\n--- Example 5: Provider Protocol ---")
            try await GenerableExamples.providerProtocolExample()

        } catch {
            print("❌ Error: \(error)")
        }
    }
}

#else
// Fallback for platforms without FoundationModels
@main
struct GenerableExamplesRunner {
    static func main() {
        print("⚠️  @Generable examples require macOS 26+ or iOS 26+ with FoundationModels framework")
        print("These examples demonstrate Apple Intelligence native structured output.")
    }
}
#endif
