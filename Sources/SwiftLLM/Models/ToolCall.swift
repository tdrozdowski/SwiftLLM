import Foundation

/// A tool call made by the LLM
public struct ToolCall: Sendable, Codable, Identifiable, Equatable {
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
