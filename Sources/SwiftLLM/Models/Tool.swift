import Foundation

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
public indirect enum ToolProperty: Sendable, Codable {
    case string(description: String?, enum: [String]?)
    case number(description: String?)
    case integer(description: String?)
    case boolean(description: String?)
    case array(items: ToolProperty, description: String?)
    case object(properties: [String: ToolProperty], description: String?)

    // Internal representation for Codable
    private struct CodableRepresentation: Codable {
        let type: String
        let description: String?
        let `enum`: [String]?
        let items: ToolProperty?
        let properties: [String: ToolProperty]?
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rep = try container.decode(CodableRepresentation.self)

        switch rep.type {
        case "string":
            self = .string(description: rep.description, enum: rep.enum)
        case "number":
            self = .number(description: rep.description)
        case "integer":
            self = .integer(description: rep.description)
        case "boolean":
            self = .boolean(description: rep.description)
        case "array":
            guard let items = rep.items else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Array type must have items"
                )
            }
            self = .array(items: items, description: rep.description)
        case "object":
            guard let properties = rep.properties else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Object type must have properties"
                )
            }
            self = .object(properties: properties, description: rep.description)
        default:
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unknown type: \(rep.type)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch self {
        case .string(let description, let enumValues):
            try container.encode(CodableRepresentation(
                type: "string",
                description: description,
                enum: enumValues,
                items: nil,
                properties: nil
            ))
        case .number(let description):
            try container.encode(CodableRepresentation(
                type: "number",
                description: description,
                enum: nil,
                items: nil,
                properties: nil
            ))
        case .integer(let description):
            try container.encode(CodableRepresentation(
                type: "integer",
                description: description,
                enum: nil,
                items: nil,
                properties: nil
            ))
        case .boolean(let description):
            try container.encode(CodableRepresentation(
                type: "boolean",
                description: description,
                enum: nil,
                items: nil,
                properties: nil
            ))
        case .array(let items, let description):
            try container.encode(CodableRepresentation(
                type: "array",
                description: description,
                enum: nil,
                items: items,
                properties: nil
            ))
        case .object(let properties, let description):
            try container.encode(CodableRepresentation(
                type: "object",
                description: description,
                enum: nil,
                items: nil,
                properties: properties
            ))
        }
    }

    // MARK: - Convenience Accessors

    var type: String {
        switch self {
        case .string: return "string"
        case .number: return "number"
        case .integer: return "integer"
        case .boolean: return "boolean"
        case .array: return "array"
        case .object: return "object"
        }
    }

    var description: String? {
        switch self {
        case .string(let description, _),
             .number(let description),
             .integer(let description),
             .boolean(let description),
             .array(_, let description),
             .object(_, let description):
            return description
        }
    }

    var properties: [String: ToolProperty]? {
        if case .object(let properties, _) = self {
            return properties
        }
        return nil
    }

    var items: ToolProperty? {
        if case .array(let items, _) = self {
            return items
        }
        return nil
    }

}
