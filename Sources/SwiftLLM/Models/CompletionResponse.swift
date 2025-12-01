import Foundation

/// Response from an LLM completion request
public struct CompletionResponse: Sendable, Equatable {
    /// The generated text (empty string if only tool calls were returned without text)
    public let text: String

    /// Tool calls requested by the model (empty if no tools used)
    public let toolCalls: [ToolCall]

    /// Model that generated the response
    public let model: String

    /// Token usage information
    public let usage: TokenUsage

    /// Finish reason (e.g., "stop", "length", "tool_calls", "content_filter")
    public let finishReason: String?

    /// Provider-specific metadata
    public let metadata: [String: String]?

    /// Whether this response requires tool execution
    public var requiresToolExecution: Bool {
        !toolCalls.isEmpty
    }

    public init(
        text: String,
        model: String,
        usage: TokenUsage,
        finishReason: String? = nil,
        metadata: [String: String]? = nil,
        toolCalls: [ToolCall] = []
    ) {
        self.text = text
        self.model = model
        self.usage = usage
        self.finishReason = finishReason
        self.metadata = metadata
        self.toolCalls = toolCalls
    }
}

/// Token usage information
public struct TokenUsage: Sendable, Equatable, Codable {
    /// Tokens in the prompt
    public let inputTokens: Int

    /// Tokens in the completion
    public let outputTokens: Int

    /// Total tokens used (input + output)
    public var totalTokens: Int {
        inputTokens + outputTokens
    }

    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }
}
