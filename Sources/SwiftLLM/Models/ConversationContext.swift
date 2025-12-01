import Foundation

/// Represents a message in a conversation
public struct ConversationMessage: Sendable, Codable, Equatable {
    public enum Role: String, Sendable, Codable {
        case user
        case assistant
        case tool
        case system
    }

    public let role: Role
    public let content: String?
    public let toolCalls: [ToolCall]?  // Optional for semantic clarity
    public let toolResults: [ToolResult]?

    public init(
        role: Role,
        content: String?,
        toolCalls: [ToolCall]? = nil,
        toolResults: [ToolResult]? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolResults = toolResults
    }
}

/// Encapsulates the state of an ongoing conversation with tool calls.
///
/// - Important: `ConversationContext` is a value type (struct). Mutations create a new copy.
///   Always capture the result of mutating operations:
///   ```swift
///   var context = ConversationContext(...)
///   context.addAssistantResponse(text: response.text, toolCalls: response.toolCalls)
///   // context now contains the updated messages
///   ```
///
///   When passing across async boundaries, be aware that each task gets a copy.
///   For shared mutable state, consider using `ConversationManager` actor instead.
public struct ConversationContext: Sendable, Codable {
    /// All messages in the conversation (provider-agnostic format)
    public internal(set) var messages: [ConversationMessage]

    /// Tools available in this conversation
    public let tools: [Tool]

    /// System prompt for the conversation
    public let systemPrompt: String?

    /// Whether the conversation is waiting for tool results
    public var requiresToolExecution: Bool {
        guard let lastMessage = messages.last else { return false }
        return lastMessage.role == .assistant && !(lastMessage.toolCalls?.isEmpty ?? true)
    }

    public init(
        systemPrompt: String?,
        tools: [Tool],
        initialMessage: String
    ) {
        self.systemPrompt = systemPrompt
        self.tools = tools
        self.messages = [ConversationMessage(role: .user, content: initialMessage)]
    }

    /// Add an assistant response with optional tool calls
    public mutating func addAssistantResponse(text: String?, toolCalls: [ToolCall]) {
        messages.append(ConversationMessage(
            role: .assistant,
            content: text,
            toolCalls: toolCalls.isEmpty ? nil : toolCalls
        ))
    }

    /// Add tool results
    public mutating func addToolResults(_ results: [ToolResult]) {
        messages.append(ConversationMessage(
            role: .tool,
            content: nil,
            toolResults: results
        ))
    }

    /// Add a user message
    public mutating func addUserMessage(_ content: String) {
        messages.append(ConversationMessage(role: .user, content: content))
    }
}

/// Actor-based conversation manager for thread-safe multi-turn conversations.
///
/// Use this when you need shared mutable conversation state across async tasks.
public actor ConversationManager {
    private var context: ConversationContext

    public init(context: ConversationContext) {
        self.context = context
    }

    public func addAssistantResponse(text: String?, toolCalls: [ToolCall]) {
        context.addAssistantResponse(text: text, toolCalls: toolCalls)
    }

    public func addToolResults(_ results: [ToolResult]) {
        context.addToolResults(results)
    }

    public func addUserMessage(_ content: String) {
        context.addUserMessage(content)
    }

    public var currentContext: ConversationContext {
        context
    }

    public var requiresToolExecution: Bool {
        context.requiresToolExecution
    }
}
