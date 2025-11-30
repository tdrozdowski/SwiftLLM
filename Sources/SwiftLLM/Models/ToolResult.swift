import Foundation

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
