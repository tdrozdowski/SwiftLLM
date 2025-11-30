import Foundation

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
public enum OpenAIToolChoice: Codable, Sendable {
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

    public init(from decoder: Decoder) throws {
        // Try to decode as a string first
        if let container = try? decoder.singleValueContainer(),
           let string = try? container.decode(String.self) {
            self = .string(string)
            return
        }

        // Otherwise decode as an object with type and function
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "function" else {
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Expected 'function' type"
            )
        }
        let functionSpec = try container.decode(FunctionSpec.self, forKey: .function)
        self = .function(name: functionSpec.name)
    }

    private enum CodingKeys: String, CodingKey {
        case type, function
    }

    private struct FunctionSpec: Codable, Sendable {
        let name: String
    }
}
