# Tool Calling Implementation Plan

## Overview

Add unified tool/function calling support across all providers (Anthropic, OpenAI, xAI, Local LLMs).

**Status**: Architecturally reviewed and updated with recommendations from principal-architect.

## Current State

- `LLMProvider` protocol has no tool calling methods
- API clients don't include tool definitions in requests
- `LLMCapabilities.supportsToolCalling` exists but isn't backed by implementation

## Architectural Principles

This implementation follows these key principles:

1. **Backward Compatibility**: Use protocol extensions with default implementations to avoid breaking existing code
2. **Type Safety**: Leverage Swift's type system to prevent runtime errors
3. **State Management**: Explicit conversation context tracking for multi-turn interactions
4. **Provider Abstraction**: Unified API that handles provider-specific format differences internally
5. **Security**: Input validation, error handling, and capability restrictions

## Provider API Differences

### OpenAI / xAI (OpenAI-compatible)
```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        },
        "required": ["location"]
      }
    }
  }],
  "tool_choice": "auto" | "none" | {"type": "function", "function": {"name": "..."}}
}

// Response includes:
"tool_calls": [{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\": \"Boston\"}"
  }
}]
```

### Anthropic Claude
```json
{
  "tools": [{
    "name": "get_weather",
    "description": "Get current weather",
    "input_schema": {
      "type": "object",
      "properties": {
        "location": { "type": "string" }
      },
      "required": ["location"]
    }
  }],
  "tool_choice": {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
}

// Response content block:
{
  "type": "tool_use",
  "id": "toolu_abc123",
  "name": "get_weather",
  "input": {"location": "Boston"}
}
```

### Local LLMs (Ollama, etc.)
- Most support OpenAI-compatible tool format
- Some may not support tools at all (handled by `supportsToolCalling` capability)

## Proposed Design

### 1. Core Tool Types (`Sources/SwiftLLM/Models/Tool.swift`)

```swift
/// Definition of a tool the LLM can call
public struct Tool: Sendable, Codable {
    public let name: String
    public let description: String
    public let parameters: ToolParameters

    public init(name: String, description: String, parameters: ToolParameters) throws {
        // Validate tool name (snake_case, no spaces) using CharacterSet for performance
        guard !name.isEmpty,
              name.first?.isLowercase == true,
              name.unicodeScalars.allSatisfy({ Self.validToolNameCharacters.contains($0) }) else {
            throw LLMError.invalidRequest(
                "Tool name '\(name)' must be snake_case starting with a lowercase letter"
            )
        }

        // Validate description length for optimal model performance
        guard description.count >= 10 && description.count <= 500 else {
            throw LLMError.invalidRequest(
                "Tool description should be between 10-500 characters for optimal model performance"
            )
        }

        // Validate that all required fields exist in properties
        if let required = parameters.required {
            for field in required {
                guard parameters.properties.keys.contains(field) else {
                    throw LLMError.invalidRequest(
                        "Required field '\(field)' not found in tool '\(name)' properties"
                    )
                }
            }
        }

        // Validate schema depth to prevent stack overflow
        try Self.validateSchemaDepth(parameters.properties, maxDepth: 10)

        self.name = name
        self.description = description
        self.parameters = parameters
    }

    // MARK: - Validation Helpers

    private static let validToolNameCharacters = CharacterSet.lowercaseLetters
        .union(.decimalDigits)
        .union(CharacterSet(charactersIn: "_"))

    private static func validateSchemaDepth(
        _ properties: [String: ToolProperty],
        depth: Int = 0,
        maxDepth: Int
    ) throws {
        guard depth < maxDepth else {
            throw LLMError.invalidRequest(
                "Tool schema exceeds maximum nesting depth of \(maxDepth)"
            )
        }
        for (_, property) in properties {
            if let nested = property.properties {
                try validateSchemaDepth(nested, depth: depth + 1, maxDepth: maxDepth)
            }
            if let items = property.items, let itemProps = items.properties {
                try validateSchemaDepth(itemProps, depth: depth + 1, maxDepth: maxDepth)
            }
        }
    }
}

/// JSON Schema for tool parameters
public struct ToolParameters: Sendable, Codable {
    public let type: String // Always "object"
    public let properties: [String: ToolProperty]
    public let required: [String]?

    public init(properties: [String: ToolProperty], required: [String]? = nil) {
        self.type = "object"
        self.properties = properties
        self.required = required
    }
}

/// Individual parameter property with support for nested objects and arrays
public struct ToolProperty: Sendable, Codable {
    public let type: String // "string", "number", "boolean", "integer", "array", "object"
    public let description: String?
    public let `enum`: [String]?
    public let items: ToolProperty? // For array types
    public let properties: [String: ToolProperty]? // For nested object types

    private init(
        type: String,
        description: String? = nil,
        enum: [String]? = nil,
        items: ToolProperty? = nil,
        properties: [String: ToolProperty]? = nil
    ) {
        self.type = type
        self.description = description
        self.enum = `enum`
        self.items = items
        self.properties = properties
    }

    // MARK: - Static Factory Methods

    public static func string(description: String? = nil, enum: [String]? = nil) -> ToolProperty {
        ToolProperty(type: "string", description: description, enum: `enum`)
    }

    public static func number(description: String? = nil) -> ToolProperty {
        ToolProperty(type: "number", description: description)
    }

    public static func integer(description: String? = nil) -> ToolProperty {
        ToolProperty(type: "integer", description: description)
    }

    public static func boolean(description: String? = nil) -> ToolProperty {
        ToolProperty(type: "boolean", description: description)
    }

    public static func array(items: ToolProperty, description: String? = nil) -> ToolProperty {
        ToolProperty(type: "array", description: description, items: items)
    }

    public static func object(
        properties: [String: ToolProperty],
        description: String? = nil
    ) -> ToolProperty {
        ToolProperty(type: "object", description: description, properties: properties)
    }
}

/// How the LLM should choose tools (provider-agnostic)
public enum ToolChoice: Sendable, Equatable, Codable {
    /// Let the model decide whether to use tools
    case auto

    /// Never use tools (text-only response)
    case none

    /// Must use at least one tool (error if model doesn't)
    case required

    /// Must use this specific tool by name
    case specific(String)

    // MARK: - Codable Conformance

    private enum CodingKeys: String, CodingKey {
        case type, name
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "auto": self = .auto
        case "none": self = .none
        case "required": self = .required
        case "specific":
            let name = try container.decode(String.self, forKey: .name)
            self = .specific(name)
        default:
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [CodingKeys.type],
                    debugDescription: "Unknown tool choice type: \(type)"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .auto: try container.encode("auto", forKey: .type)
        case .none: try container.encode("none", forKey: .type)
        case .required: try container.encode("required", forKey: .type)
        case .specific(let name):
            try container.encode("specific", forKey: .type)
            try container.encode(name, forKey: .name)
        }
    }

    // MARK: - Provider-Specific Conversion

    /// Convert to OpenAI/xAI format (type-safe)
    func toOpenAIFormat() -> OpenAIToolChoice {
        switch self {
        case .auto: return .string("auto")
        case .none: return .string("none")
        case .required: return .string("required")
        case .specific(let name): return .function(name: name)
        }
    }

    /// Convert to Anthropic format
    func toAnthropicFormat() -> [String: Any] {
        switch self {
        case .auto: return ["type": "auto"]
        case .none: return ["type": "auto"] // Anthropic doesn't have "none", just omit tools
        case .required: return ["type": "any"]
        case .specific(let name): return ["type": "tool", "name": name]
        }
    }
}

/// Type-safe OpenAI tool choice representation
public enum OpenAIToolChoice: Encodable {
    case string(String)  // "auto", "none", "required"
    case function(name: String)

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .string(let value):
            var container = encoder.singleValueContainer()
            try container.encode(value)
        case .function(let name):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("function", forKey: .type)
            try container.encode(FunctionSpec(name: name), forKey: .function)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type, function
    }

    private struct FunctionSpec: Encodable {
        let name: String
    }
}

/// A tool call made by the LLM
public struct ToolCall: Sendable, Codable, Identifiable {
    public let id: String
    public let name: String
    public let arguments: String // JSON string

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// Decode arguments to a specific type
    public func decodeArguments<T: Decodable>(_ type: T.Type) throws -> T {
        guard let data = arguments.data(using: .utf8) else {
            throw LLMError.decodingError(
                "Invalid UTF-8 encoding in arguments for tool '\(name)' (id: \(id))"
            )
        }
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch let decodingError as DecodingError {
            let context = Self.describeDecodingError(decodingError)
            throw LLMError.decodingError(
                "Failed to decode arguments for tool '\(name)': \(context)\nRaw JSON: \(arguments)"
            )
        } catch {
            throw LLMError.decodingError(
                "Failed to decode arguments for tool '\(name)': \(error.localizedDescription)"
            )
        }
    }

    /// Provide detailed context for decoding errors
    private static func describeDecodingError(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, let context):
            let path = context.codingPath.map(\.stringValue).joined(separator: ".")
            return "Missing key '\(key.stringValue)'\(path.isEmpty ? "" : " at path: \(path)")"
        case .typeMismatch(let type, let context):
            let path = context.codingPath.map(\.stringValue).joined(separator: ".")
            return "Type mismatch for \(type)\(path.isEmpty ? "" : " at path: \(path)")"
        case .valueNotFound(let type, let context):
            let path = context.codingPath.map(\.stringValue).joined(separator: ".")
            return "Null value for \(type)\(path.isEmpty ? "" : " at path: \(path)")"
        case .dataCorrupted(let context):
            return "Corrupted data: \(context.debugDescription)"
        @unknown default:
            return error.localizedDescription
        }
    }
}

/// Structured error information for tool execution failures
public struct ToolExecutionError: Error, Sendable, Codable, Equatable {
    public enum Category: String, Sendable, Codable {
        case invalidArguments
        case authenticationFailed
        case rateLimited
        case resourceNotFound
        case executionTimeout
        case networkError
        case permissionDenied
        case cancelled
        case unknown
    }

    public let category: Category
    public let message: String
    public let details: [String: String]?

    public init(category: Category, message: String, details: [String: String]? = nil) {
        self.category = category
        self.message = message
        self.details = details
    }

    /// Convert to a format suitable for the LLM
    public func toLLMFormat() -> String {
        var errorMessage = "Tool execution failed (\(category.rawValue)): \(message)"
        if let details = details, !details.isEmpty {
            let detailsStr = details.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
            errorMessage += "\nDetails: \(detailsStr)"
        }
        return errorMessage
    }
}

/// Result of executing a tool
public struct ToolResult: Sendable, Codable, Equatable {
    public let toolCallId: String
    public let result: ResultContent

    public enum ResultContent: Sendable, Codable, Equatable {
        case success(String) // JSON or plain text
        case error(ToolExecutionError)

        // MARK: - Manual Codable Implementation

        private enum CodingKeys: String, CodingKey {
            case type, content, error
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            switch type {
            case "success":
                let content = try container.decode(String.self, forKey: .content)
                self = .success(content)
            case "error":
                let error = try container.decode(ToolExecutionError.self, forKey: .error)
                self = .error(error)
            default:
                throw DecodingError.dataCorrupted(
                    DecodingError.Context(
                        codingPath: [CodingKeys.type],
                        debugDescription: "Unknown result type: \(type)"
                    )
                )
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .success(let content):
                try container.encode("success", forKey: .type)
                try container.encode(content, forKey: .content)
            case .error(let error):
                try container.encode("error", forKey: .type)
                try container.encode(error, forKey: .error)
            }
        }
    }

    public var isError: Bool {
        if case .error = result { return true }
        return false
    }

    public init(toolCallId: String, result: ResultContent) {
        self.toolCallId = toolCallId
        self.result = result
    }

    /// Create from Encodable result
    public static func success<T: Encodable>(id: String, _ value: T) throws -> ToolResult {
        let data = try JSONEncoder().encode(value)
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return ToolResult(toolCallId: id, result: .success(json))
    }

    /// Create error result
    public static func error(id: String, error: ToolExecutionError) -> ToolResult {
        ToolResult(toolCallId: id, result: .error(error))
    }

    /// Create error result from message
    public static func error(id: String, message: String) -> ToolResult {
        let error = ToolExecutionError(category: .unknown, message: message)
        return ToolResult(toolCallId: id, result: .error(error))
    }
}
```

### 2. Conversation Context (`Sources/SwiftLLM/Models/ConversationContext.swift`)

**Critical**: Proper state management for multi-turn tool conversations requires tracking full message history.

```swift
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
```

### 3. Extended Response Type

**Breaking Change Mitigation**: Extend existing `CompletionResponse` while maintaining backward compatibility.

```swift
// In Sources/SwiftLLM/Models/CompletionResponse.swift
// MODIFY existing struct:

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
        toolCalls: [ToolCall] = []  // NEW: defaults to empty for backward compatibility
    ) {
        self.text = text
        self.model = model
        self.usage = usage
        self.finishReason = finishReason
        self.metadata = metadata
        self.toolCalls = toolCalls
    }
}
```

**Note**: `text` remains non-optional to maintain true backward compatibility. When a model returns only tool calls without accompanying text, providers should set `text` to an empty string. Check `requiresToolExecution` to determine if tool execution is needed.

### 4. Protocol Extension with Default Implementations

**Critical**: Use protocol extensions to avoid breaking existing implementations.

```swift
// In Sources/SwiftLLM/Protocols/LLMProvider.swift

// Add these methods to the LLMProvider protocol:
public protocol LLMProvider: Sendable {
    // ... existing methods ...

    /// Generate completion with tool support
    /// - Note: Default implementation throws `.unsupportedFeature` if not overridden
    func generateCompletionWithTools(
        context: ConversationContext,
        toolChoice: ToolChoice,
        options: GenerationOptions
    ) async throws -> CompletionResponse

    /// Continue conversation after tool execution
    /// - Note: Default implementation throws `.unsupportedFeature` if not overridden
    func continueWithToolResults(
        context: ConversationContext,
        options: GenerationOptions
    ) async throws -> CompletionResponse
}

// Add default implementations in extension:
public extension LLMProvider {
    func generateCompletionWithTools(
        context: ConversationContext,
        toolChoice: ToolChoice = .auto,
        options: GenerationOptions = .default
    ) async throws -> CompletionResponse {
        guard capabilities.supportsToolCalling else {
            throw LLMError.unsupportedFeature("Tool calling is not supported by \(displayName)")
        }
        // Default implementation for providers that haven't implemented tool support yet
        throw LLMError.unsupportedFeature("Tool calling not yet implemented for \(displayName)")
    }

    func continueWithToolResults(
        context: ConversationContext,
        options: GenerationOptions = .default
    ) async throws -> CompletionResponse {
        guard capabilities.supportsToolCalling else {
            throw LLMError.unsupportedFeature("Tool calling is not supported by \(displayName)")
        }
        throw LLMError.unsupportedFeature("Tool calling not yet implemented for \(displayName)")
    }
}
```

### 5. Parallel Tool Execution Helper

Support for executing multiple tool calls concurrently using Swift's structured concurrency.

```swift
// In Sources/SwiftLLM/Utilities/ToolExecutor.swift (optional helper)

/// Execute multiple tool calls concurrently with cancellation support
public func executeToolCalls(
    _ calls: [ToolCall],
    using executor: @Sendable @escaping (ToolCall) async throws -> ToolResult
) async -> [ToolResult] {
    await withTaskGroup(of: (String, ToolResult).self) { group in
        for call in calls {
            group.addTask {
                // Check for cancellation before executing
                if Task.isCancelled {
                    return (call.id, .error(
                        id: call.id,
                        error: ToolExecutionError(category: .cancelled, message: "Task cancelled")
                    ))
                }

                do {
                    let result = try await executor(call)
                    return (call.id, result)
                } catch is CancellationError {
                    return (call.id, .error(
                        id: call.id,
                        error: ToolExecutionError(category: .cancelled, message: "Task cancelled")
                    ))
                } catch let error as ToolExecutionError {
                    return (call.id, .error(id: call.id, error: error))
                } catch {
                    let execError = ToolExecutionError(
                        category: .unknown,
                        message: error.localizedDescription
                    )
                    return (call.id, .error(id: call.id, error: execError))
                }
            }
        }

        var results: [String: ToolResult] = [:]
        for await (id, result) in group {
            results[id] = result
        }

        // Preserve original order of tool calls
        return calls.compactMap { results[$0.id] }
    }
}
```

### 6. Tool Builder DSL (Optional, Phase 4+)

```swift
// Fluent builder for tools
let weatherTool = Tool.build("get_weather") {
    $0.description("Get current weather for a location")
    $0.parameter("location", type: .string, required: true) {
        $0.description("City name")
    }
    $0.parameter("unit", type: .string) {
        $0.description("Temperature unit")
        $0.enum(["celsius", "fahrenheit"])
    }
}
```

### 7. Type-Safe Tool Protocol (Optional, Phase 4+)

```swift
/// Protocol for type-safe tool definitions with automatic schema generation
///
/// Tools implementing this protocol must ensure their `execute` method is thread-safe.
/// Tools that access shared mutable state should use actor isolation or appropriate
/// synchronization mechanisms. SwiftLLM may execute multiple tools concurrently when
/// a model requests parallel tool calls.
public protocol LLMTool: Sendable {
    associatedtype Input: Codable & Sendable
    associatedtype Output: Codable & Sendable

    static var name: String { get }
    static var description: String { get }

    /// Automatically generate Tool schema from Input type
    /// - Throws: If schema generation fails
    static var tool: Tool { get throws }

    /// Execute the tool with typed input
    /// - Parameter input: Validated and decoded input
    /// - Returns: The tool's output
    /// - Note: Implementation must be thread-safe or use actor isolation
    func execute(input: Input) async throws -> Output
}

// Example usage with actor isolation:
actor GetWeatherTool: LLMTool {
    struct Input: Codable, Sendable {
        let location: String
        let unit: String?
    }

    struct Output: Codable, Sendable {
        let temperature: Double
        let conditions: String
        let humidity: Int
    }

    static let name = "get_weather"
    static let description = "Get current weather for a location"
    static var tool: Tool {
        get throws {
            try Tool(
                name: name,
                description: description,
                parameters: ToolParameters(
                    properties: [
                        "location": .string(description: "City name, e.g., 'San Francisco, CA'"),
                        "unit": .string(description: "Temperature unit", enum: ["celsius", "fahrenheit"])
                    ],
                    required: ["location"]
                )
            )
        }
    }

    private let apiClient: WeatherAPIClient

    func execute(input: Input) async throws -> Output {
        // Actor-isolated execution
        try await apiClient.fetchWeather(
            location: input.location,
            unit: input.unit ?? "celsius"
        )
    }
}
```

## Implementation Steps

### Phase 1: Core Types & Protocol (4-5 hours)

**Goal**: Create the foundation with proper validation and backward compatibility.

1. ✅ Create `Tool.swift` with validation (name format, description length, required fields)
2. ✅ Create `ToolCall.swift` with enhanced error messages
3. ✅ Create `ToolResult.swift` with `ToolExecutionError` support
4. ✅ Create `ConversationContext.swift` and `ConversationMessage.swift`
5. ✅ Update `CompletionResponse.swift` to include optional `toolCalls` array
6. ✅ Add protocol methods to `LLMProvider` with default implementations
7. ✅ Write comprehensive unit tests for core types:
   - Tool validation (valid/invalid names, descriptions, required fields)
   - ToolCall argument decoding (valid/invalid JSON, type mismatches)
   - ConversationContext state management
   - ToolResult success/error creation

### Phase 2: OpenAI Implementation (3-4 hours)

**Goal**: Implement tool calling for OpenAI as the reference implementation.

8. ✅ Update `OpenAIAPIClient.swift`:
   - Add `tools` and `tool_choice` to request struct
   - Add `tool_calls` to response struct
   - Implement request builder pattern for complex requests
9. ✅ Implement `generateCompletionWithTools` in `OpenAIProvider.swift`:
   - Convert `ConversationContext` to OpenAI message format
   - Convert `ToolChoice` to OpenAI format
   - Parse tool calls from response
10. ✅ Implement `continueWithToolResults` in `OpenAIProvider.swift`:
    - Append tool results to message history
    - Continue conversation with provider
11. ✅ Test with real OpenAI API (gpt-4, gpt-4-turbo)
12. ✅ Add integration tests with mock API responses

### Phase 3: Anthropic & xAI (4-5 hours)

**Goal**: Extend to other major providers with different tool formats.

13. ✅ Update `AnthropicAPIClient.swift`:
    - Implement Anthropic's tool format (name, description, input_schema)
    - Handle content blocks with tool_use type
    - Convert tool results to Anthropic message format
14. ✅ Implement tool calling in `AnthropicProvider.swift`:
    - Convert between unified and Anthropic formats
    - Handle multiple content blocks
15. ✅ Update `XAIProvider.swift`:
    - Reuse OpenAI implementation (xAI is OpenAI-compatible)
    - Test with Grok models
16. ✅ Test with real Anthropic and xAI APIs
17. ✅ Add provider-specific integration tests

### Phase 4: LocalLLM & Polish (3-4 hours)

**Goal**: Complete provider coverage and add documentation/examples.

18. ✅ Update `LocalLLMProvider.swift`:
    - Implement OpenAI-compatible tool format
    - Handle cases where local model doesn't support tools
19. ✅ Create `Examples/ToolCallingExamples.swift`:
    - Simple single-tool example
    - Multi-turn conversation example
    - Parallel tool execution example
    - Type-safe LLMTool protocol example
    - Error handling example
20. ✅ Update `README.md` with tool calling section
21. ✅ Add tool calling guide to documentation
22. ✅ Create security best practices guide
23. ✅ Add end-to-end integration tests

### Phase 5: Advanced Features (Optional, 6-8 hours)

**Goal**: Type safety, tool registry, and advanced patterns.

24. ⬜ Implement automatic schema generation from Codable types
25. ⬜ Create `ToolRegistry` actor for centralized tool management
26. ⬜ Add tool execution helper utilities
27. ⬜ Add tool capability restrictions (network, filesystem, database)
28. ⬜ Implement rate limiting for tool execution
29. ⬜ Research and prototype streaming with tools (complex, may defer to Phase 6+)

**Total Estimate**:
- Phases 1-4 (MVP): **14-18 hours**
- Including Phase 5: **20-26 hours**

## Files to Create/Modify

### New Files
- `Sources/SwiftLLM/Models/Tool.swift` - Tool, ToolParameters, ToolProperty with validation
- `Sources/SwiftLLM/Models/ToolCall.swift` - ToolCall with enhanced decoding
- `Sources/SwiftLLM/Models/ToolResult.swift` - ToolResult and ToolExecutionError (with Equatable)
- `Sources/SwiftLLM/Models/ToolChoice.swift` - ToolChoice and OpenAIToolChoice
- `Sources/SwiftLLM/Models/ConversationContext.swift` - ConversationContext, ConversationMessage, ConversationManager
- `Sources/SwiftLLM/Utilities/ToolExecutor.swift` - Parallel execution helper (optional)
- `Examples/ToolCallingExamples.swift` - Comprehensive examples
- `Tests/SwiftLLMTests/ToolTests.swift` - Unit tests for tool types
- `Tests/SwiftLLMTests/ToolProviderTests.swift` - Integration tests

### Modified Files
- `Sources/SwiftLLM/Models/CompletionResponse.swift` - Add `toolCalls` property (text remains non-optional)
- `Sources/SwiftLLM/Protocols/LLMProvider.swift` - Add tool methods with defaults (note: default params only work via concrete types, not existentials)
- `Sources/SwiftLLM/Clients/OpenAIAPIClient.swift` - Add tool request/response support
- `Sources/SwiftLLM/Clients/AnthropicAPIClient.swift` - Add Anthropic tool format
- `Sources/SwiftLLM/Clients/XAIAPIClient.swift` - Add tool support (reuse OpenAI)
- `Sources/SwiftLLM/Providers/OpenAIProvider.swift` - Implement tool methods
- `Sources/SwiftLLM/Providers/AnthropicProvider.swift` - Implement tool methods
- `Sources/SwiftLLM/Providers/XAIProvider.swift` - Implement tool methods
- `Sources/SwiftLLM/Providers/LocalLLMProvider.swift` - Implement tool methods
- `README.md` - Add tool calling documentation and examples

## Design Decisions & Resolutions

### 1. Streaming with Tools ✅ DEFERRED
**Decision**: Defer to Phase 5+
**Rationale**:
- OpenAI streams tool calls incrementally with `delta` objects
- Anthropic uses content blocks which complicate streaming
- Bidirectional streaming for tool execution results is complex
- MVP should focus on request/response pattern first
**Action**: Document limitation in protocol with note for future support

### 2. Parallel Tool Calls ✅ RESOLVED
**Decision**: Support multiple concurrent tool calls
**Implementation**:
- `CompletionResponse.toolCalls` is an array (supports multiple calls)
- Provide `executeToolCalls` helper using `TaskGroup` for concurrent execution
- Preserve order of results to match order of calls
**Rationale**: OpenAI and Anthropic both support this, critical for real-world use

### 3. Tool Execution Helper ✅ RESOLVED
**Decision**: Provide optional helper, don't enforce pattern
**Implementation**:
- Create `executeToolCalls()` utility function in `ToolExecutor.swift`
- User provides closure to execute individual tools
- Framework handles concurrency, error catching, and result ordering
**Rationale**: Gives flexibility while providing convenience

### 4. Apple AFM Tool Support ❓ NEEDS RESEARCH
**Decision**: Check before implementing
**Action Items**:
1. Review FoundationModels framework documentation (macOS 26+)
2. If AFM doesn't support tools: `AppleIntelligenceProvider.capabilities.supportsToolCalling = false`
3. Default protocol extension will handle gracefully with `.unsupportedFeature` error
**Fallback**: Existing error handling ensures graceful degradation

### 5. Breaking Changes Strategy ✅ RESOLVED
**Decision**: Use backward-compatible approach
**Implementation**:
- Protocol extensions with default implementations (no breaking changes)
- Extend `CompletionResponse` with optional `toolCalls: [ToolCall] = []`
- Existing code continues to work without modifications
**Migration Path**: Users opt-in to tool calling when ready

## Example Usage (Target API)

### Example 1: Basic Tool Calling

```swift
// Define tools using static factory methods
let weatherTool = try Tool(
    name: "get_weather",
    description: "Get current weather for a location",
    parameters: ToolParameters(
        properties: [
            "location": .string(description: "City name, e.g., 'San Francisco, CA'"),
            "unit": .string(description: "Temperature unit", enum: ["celsius", "fahrenheit"])
        ],
        required: ["location"]
    )
)

// Create conversation context
var context = ConversationContext(
    systemPrompt: "You are a helpful weather assistant",
    tools: [weatherTool],
    initialMessage: "What's the weather in San Francisco?"
)

// Make request with tools
let provider = OpenAIProvider.gpt4Turbo(apiKey: apiKey)
let response = try await provider.generateCompletionWithTools(
    context: context,
    toolChoice: .auto,
    options: .default
)

// Handle tool calls
if response.requiresToolExecution {
    // Execute tools (can be done in parallel)
    var results: [ToolResult] = []

    for call in response.toolCalls {
        switch call.name {
        case "get_weather":
            do {
                let args = try call.decodeArguments(WeatherArgs.self)
                let weather = try await fetchWeather(location: args.location, unit: args.unit)
                results.append(try .success(id: call.id, weather))
            } catch {
                results.append(.error(
                    id: call.id,
                    error: ToolExecutionError(category: .networkError, message: error.localizedDescription)
                ))
            }
        default:
            results.append(.error(
                id: call.id,
                error: ToolExecutionError(category: .resourceNotFound, message: "Unknown tool '\(call.name)'")
            ))
        }
    }

    // Add results to context
    context.addAssistantResponse(text: response.text, toolCalls: response.toolCalls)
    context.addToolResults(results)

    // Continue conversation
    let finalResponse = try await provider.continueWithToolResults(
        context: context,
        options: .default
    )

    print(finalResponse.text ?? "No response")
}
```

### Example 2: Parallel Tool Execution

```swift
// Execute multiple tools concurrently
let results = await executeToolCalls(response.toolCalls) { call in
    switch call.name {
    case "get_weather":
        let args = try call.decodeArguments(WeatherArgs.self)
        let weather = try await fetchWeather(location: args.location)
        return try .success(id: call.id, weather)

    case "get_time":
        let args = try call.decodeArguments(TimeArgs.self)
        let time = try await fetchTime(timezone: args.timezone)
        return try .success(id: call.id, time)

    default:
        return .error(id: call.id, message: "Unknown tool")
    }
}
```

### Example 3: Type-Safe Tools (Advanced)

```swift
// Define a type-safe tool
actor WeatherTool: LLMTool {
    struct Input: Codable, Sendable {
        let location: String
        let unit: String?
    }

    struct Output: Codable, Sendable {
        let temperature: Double
        let conditions: String
        let humidity: Int
    }

    static let name = "get_weather"
    static let description = "Get current weather for a location"
    static var tool: Tool {
        get throws {
            try Tool(
                name: name,
                description: description,
                parameters: ToolParameters(
                    properties: [
                        "location": .string(description: "City name"),
                        "unit": .string(description: "Temperature unit", enum: ["celsius", "fahrenheit"])
                    ],
                    required: ["location"]
                )
            )
        }
    }

    func execute(input: Input) async throws -> Output {
        // Call weather API
        try await weatherAPI.fetch(location: input.location, unit: input.unit ?? "celsius")
    }
}

// Usage
let weatherTool = WeatherTool()
let context = ConversationContext(
    systemPrompt: nil,
    tools: [try WeatherTool.tool],
    initialMessage: "What's the weather in Boston?"
)
```

## Security Considerations

### Input Validation
- ✅ Tool names validated (snake_case, no special characters)
- ✅ Description length validated (10-500 characters)
- ✅ Required fields checked against properties
- ⬜ Consider adding JSON Schema validation for complex parameter types

### Tool Execution Safety
```swift
/// Define what capabilities a tool requires (Phase 5+)
public struct ToolCapabilities: OptionSet, Sendable {
    public let rawValue: Int

    public static let networkAccess = ToolCapabilities(rawValue: 1 << 0)
    public static let fileSystemAccess = ToolCapabilities(rawValue: 1 << 1)
    public static let databaseAccess = ToolCapabilities(rawValue: 1 << 2)
    public static let shellAccess = ToolCapabilities(rawValue: 1 << 3)
}
```

### Best Practices Documentation
1. **Always validate tool inputs** before execution
2. **Use allowlists** for sensitive operations (file paths, commands)
3. **Implement rate limiting** for tools that call external APIs
4. **Sanitize outputs** before returning to the model
5. **Never execute arbitrary code** from tool arguments
6. **Log tool executions** for audit trails
7. **Use actor isolation** for tools with shared state

### Prompt Injection Defense
- Warn users about tools that execute commands or access sensitive data
- Document that tool descriptions should not contain user-controllable text
- Recommend sanitizing location/file path inputs

## Testing Strategy

### Unit Tests (Required for MVP)

```swift
// ToolTests.swift
func testToolValidation() {
    // Valid tool
    XCTAssertNoThrow(try Tool(name: "valid_tool", description: "A valid description", parameters: ...))

    // Invalid name
    XCTAssertThrowsError(try Tool(name: "Invalid Tool", description: "...", parameters: ...))

    // Invalid description
    XCTAssertThrowsError(try Tool(name: "tool", description: "short", parameters: ...))

    // Missing required field
    XCTAssertThrowsError(try Tool(name: "tool", description: "...", parameters: ToolParameters(
        properties: ["foo": .string()],
        required: ["bar"] // 'bar' not in properties
    )))
}

func testToolCallDecoding() {
    let call = ToolCall(id: "1", name: "test", arguments: #"{"value": 42}"#)
    XCTAssertNoThrow(try call.decodeArguments(TestArgs.self))

    let invalid = ToolCall(id: "2", name: "test", arguments: "invalid json")
    XCTAssertThrowsError(try invalid.decodeArguments(TestArgs.self))
}

func testConversationContext() {
    var context = ConversationContext(
        systemPrompt: "test",
        tools: [],
        initialMessage: "hello"
    )

    XCTAssertEqual(context.messages.count, 1)
    XCTAssertFalse(context.requiresToolExecution)

    context.addAssistantResponse(text: nil, toolCalls: [ToolCall(...)])
    XCTAssertTrue(context.requiresToolExecution)
}
```

### Integration Tests

```swift
// ToolProviderTests.swift
func testOpenAIToolCalling() async throws {
    // Use mock API client
    let response = try await provider.generateCompletionWithTools(
        context: testContext,
        toolChoice: .auto,
        options: .default
    )

    XCTAssertFalse(response.toolCalls.isEmpty)
    XCTAssertEqual(response.finishReason, "tool_calls")
}

func testAnthropicFormatConversion() {
    // Test conversion between unified and Anthropic format
    let tool = try Tool(...)
    let anthropicFormat = convertToAnthropicFormat(tool)
    XCTAssertEqual(anthropicFormat["name"], tool.name)
    XCTAssertNotNil(anthropicFormat["input_schema"])
}
```

### End-to-End Tests

```swift
func testMultiTurnToolConversation() async throws {
    // Test complete flow: request -> tool execution -> continuation
    var context = ConversationContext(...)

    let response1 = try await provider.generateCompletionWithTools(context: context, ...)
    XCTAssertTrue(response1.requiresToolExecution)

    let results = [/* execute tools */]
    context.addAssistantResponse(text: response1.text, toolCalls: response1.toolCalls)
    context.addToolResults(results)

    let response2 = try await provider.continueWithToolResults(context: context, ...)
    XCTAssertFalse(response2.requiresToolExecution)
    XCTAssertNotNil(response2.text)
}
```

## Performance Considerations

### Potential Bottlenecks
1. **JSON Encoding/Decoding**: Acceptable for MVP, optimize later if needed
2. **Network Round-Trips**: Minimum 2 API calls per tool interaction
3. **Large Tool Schemas**: Keep tool descriptions concise (10-20 tools max per request)

### Optimization Opportunities (Phase 5+)
```swift
/// Cache parsed tool definitions
actor ToolCache {
    private var cache: [String: Tool] = [:]

    func getCached(name: String, generator: () throws -> Tool) rethrows -> Tool {
        if let cached = cache[name] {
            return cached
        }
        let tool = try generator()
        cache[name] = tool
        return tool
    }
}
```

## Migration Guide

### For Existing Users

**No breaking changes!** Existing code continues to work without modifications.

To adopt tool calling:

```swift
// Before (still works)
let response = try await provider.generateCompletion(
    prompt: "Hello",
    systemPrompt: nil,
    options: .default
)
// response.text is always a String

// After (opt-in to tools)
var context = ConversationContext(
    systemPrompt: nil,
    tools: [myTool],
    initialMessage: "Hello"
)
let response = try await provider.generateCompletionWithTools(
    context: context,
    toolChoice: .auto,
    options: .default
)
// response.text is still a String (empty if only tool calls)
// response.toolCalls contains any tool calls
if response.requiresToolExecution {
    // Handle tool execution
}
```

### Important Notes

1. **Default Parameters and Protocol Types**: Default parameter values in protocol extensions only work when calling through concrete types, not protocol-typed variables:
   ```swift
   // Works - defaults available:
   let provider = OpenAIProvider.gpt4Turbo(apiKey: key)
   let response = try await provider.generateCompletionWithTools(context: ctx)

   // Doesn't compile - must specify all parameters:
   let provider: any LLMProvider = OpenAIProvider.gpt4Turbo(apiKey: key)
   let response = try await provider.generateCompletionWithTools(
       context: ctx,
       toolChoice: .auto,  // Required
       options: .default    // Required
   )
   ```

2. **ConversationContext Copy Semantics**: `ConversationContext` is a struct (value type). Mutations create copies:
   ```swift
   var context = ConversationContext(...)
   context.addAssistantResponse(text: "Hi", toolCalls: [])
   // context now has the new message

   // For shared mutable state across async tasks, use ConversationManager:
   let manager = ConversationManager(context: context)
   await manager.addAssistantResponse(text: "Hi", toolCalls: [])
   ```

3. **Empty Text Field**: When a model returns only tool calls without text, `response.text` will be an empty string (not nil). Always check `response.requiresToolExecution` to determine if tool handling is needed.

### CHANGELOG Entry

```markdown
## [Unreleased]

### Added
- **Tool Calling Support**: Unified tool/function calling across all providers
  - Anthropic Claude (all models)
  - OpenAI (GPT-4, GPT-4 Turbo, GPT-3.5 Turbo)
  - xAI Grok (all models)
  - Local LLMs via Ollama/LM Studio (if supported by model)
- New types: `Tool`, `ToolCall`, `ToolResult`, `ToolExecutionError`
- `ConversationContext` for multi-turn tool conversations
- `LLMTool` protocol for type-safe tool definitions (optional)
- Parallel tool execution helper
- Comprehensive examples and documentation

### Changed
- `CompletionResponse` now includes `toolCalls: [ToolCall]` property (backward compatible, defaults to empty array)
  - `text` remains non-optional `String` (empty string when only tool calls returned)
- `LLMProvider` protocol extended with `generateCompletionWithTools` and `continueWithToolResults`
  - Default implementations provided (no breaking changes)
  - Graceful error for providers that don't support tools
  - Note: Default parameter values only available when calling via concrete types, not protocol existentials

### Security
- Tool input validation (name format, description length, schema validation)
- Structured error handling with `ToolExecutionError`
- Documentation on secure tool implementation

### Performance
- Concurrent tool execution using Swift's structured concurrency
- Efficient message history management

### Migration Guide
No changes required for existing code. To use tool calling, see the new `Examples/ToolCallingExamples.swift`.
```
