import XCTest
@testable import SwiftLLM

final class ToolTests: XCTestCase {

    // MARK: - Tool Validation Tests

    func testToolWithValidName() throws {
        let tool = try Tool(
            name: "get_weather",
            description: "Get current weather for a location",
            parameters: ToolParameters(properties: [:])
        )
        XCTAssertEqual(tool.name, "get_weather")
    }

    func testToolWithInvalidNameThrows() {
        // Invalid: contains spaces
        XCTAssertThrowsError(try Tool(
            name: "get weather",
            description: "Get current weather for a location",
            parameters: ToolParameters(properties: [:])
        ))

        // Invalid: starts with uppercase
        XCTAssertThrowsError(try Tool(
            name: "GetWeather",
            description: "Get current weather for a location",
            parameters: ToolParameters(properties: [:])
        ))

        // Invalid: contains special characters
        XCTAssertThrowsError(try Tool(
            name: "get-weather",
            description: "Get current weather for a location",
            parameters: ToolParameters(properties: [:])
        ))

        // Invalid: empty string
        XCTAssertThrowsError(try Tool(
            name: "",
            description: "Get current weather for a location",
            parameters: ToolParameters(properties: [:])
        ))
    }

    func testToolWithInvalidDescriptionThrows() {
        // Too short
        XCTAssertThrowsError(try Tool(
            name: "get_weather",
            description: "Short",
            parameters: ToolParameters(properties: [:])
        ))

        // Too long (>500 characters)
        let longDescription = String(repeating: "a", count: 501)
        XCTAssertThrowsError(try Tool(
            name: "get_weather",
            description: longDescription,
            parameters: ToolParameters(properties: [:])
        ))
    }

    func testToolWithMissingRequiredFieldThrows() {
        XCTAssertThrowsError(try Tool(
            name: "get_weather",
            description: "Get current weather for a location",
            parameters: ToolParameters(
                properties: ["location": .string(description: "City name", enum: nil)],
                required: ["location", "unit"] // "unit" doesn't exist
            )
        ))
    }

    func testToolWithValidRequiredFields() throws {
        let tool = try Tool(
            name: "get_weather",
            description: "Get current weather for a location",
            parameters: ToolParameters(
                properties: [
                    "location": .string(description: "City name", enum: nil),
                    "unit": .string(description: "Temperature unit", enum: ["celsius", "fahrenheit"])
                ],
                required: ["location"]
            )
        )
        XCTAssertEqual(tool.parameters.required, ["location"])
    }

    func testToolWithDeeplyNestedSchemaThrows() {
        // Create a deeply nested schema (>10 levels)
        var property: ToolProperty = .string(description: "Deep value", enum: nil)
        for _ in 0..<11 {
            property = .object(properties: ["nested": property], description: nil)
        }

        XCTAssertThrowsError(try Tool(
            name: "nested_tool",
            description: "A tool with deeply nested schema",
            parameters: ToolParameters(properties: ["root": property])
        ))
    }

    // MARK: - ToolProperty Tests

    func testToolPropertyTypes() {
        let stringProp = ToolProperty.string(description: "A string", enum: nil)
        XCTAssertEqual(stringProp.type, "string")

        let numberProp = ToolProperty.number(description: "A number")
        XCTAssertEqual(numberProp.type, "number")

        let integerProp = ToolProperty.integer(description: "An integer")
        XCTAssertEqual(integerProp.type, "integer")

        let booleanProp = ToolProperty.boolean(description: "A boolean")
        XCTAssertEqual(booleanProp.type, "boolean")

        let arrayProp = ToolProperty.array(
            items: .string(description: nil, enum: nil),
            description: "An array"
        )
        XCTAssertEqual(arrayProp.type, "array")

        let objectProp = ToolProperty.object(
            properties: ["key": .string(description: nil, enum: nil)],
            description: "An object"
        )
        XCTAssertEqual(objectProp.type, "object")
    }

    func testToolPropertyCodable() throws {
        let property = ToolProperty.object(
            properties: [
                "name": .string(description: "Person's name", enum: nil),
                "age": .integer(description: "Person's age"),
                "tags": .array(items: .string(description: nil, enum: nil), description: "Tags")
            ],
            description: "A person object"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(property)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ToolProperty.self, from: data)

        XCTAssertEqual(decoded.type, "object")
        XCTAssertEqual(decoded.properties?.count, 3)
    }

    // MARK: - ToolCall Tests

    func testToolCallDecoding() throws {
        struct TestArgs: Codable {
            let location: String
            let unit: String?
        }

        let call = ToolCall(
            id: "call_123",
            name: "get_weather",
            arguments: #"{"location": "San Francisco", "unit": "celsius"}"#
        )

        let args = try call.decodeArguments(TestArgs.self)
        XCTAssertEqual(args.location, "San Francisco")
        XCTAssertEqual(args.unit, "celsius")
    }

    func testToolCallDecodingInvalidJSON() {
        struct TestArgs: Codable {
            let value: Int
        }

        let call = ToolCall(
            id: "call_123",
            name: "test",
            arguments: "invalid json"
        )

        XCTAssertThrowsError(try call.decodeArguments(TestArgs.self))
    }

    func testToolCallDecodingTypeMismatch() {
        struct TestArgs: Codable {
            let value: Int
        }

        let call = ToolCall(
            id: "call_123",
            name: "test",
            arguments: #"{"value": "not_an_int"}"#
        )

        XCTAssertThrowsError(try call.decodeArguments(TestArgs.self))
    }

    // MARK: - ToolResult Tests

    func testToolResultSuccess() throws {
        struct WeatherResponse: Codable {
            let temperature: Double
            let conditions: String
        }

        let weather = WeatherResponse(temperature: 72.5, conditions: "Sunny")
        let result = try ToolResult.success(id: "call_123", weather)

        XCTAssertFalse(result.isError)
        XCTAssertEqual(result.toolCallId, "call_123")
    }

    func testToolResultError() {
        let error = ToolExecutionError(
            category: .networkError,
            message: "Connection timeout"
        )
        let result = ToolResult.error(id: "call_123", error: error)

        XCTAssertTrue(result.isError)
        XCTAssertEqual(result.toolCallId, "call_123")
    }

    func testToolResultErrorFromMessage() {
        let result = ToolResult.error(id: "call_123", message: "Something went wrong")

        XCTAssertTrue(result.isError)
        if case .error(let error) = result.result {
            XCTAssertEqual(error.category, .unknown)
            XCTAssertEqual(error.message, "Something went wrong")
        } else {
            XCTFail("Expected error result")
        }
    }

    func testToolExecutionErrorToLLMFormat() {
        let error = ToolExecutionError(
            category: .rateLimited,
            message: "Too many requests",
            details: ["retry_after": "60"]
        )

        let formatted = error.toLLMFormat()
        XCTAssertTrue(formatted.contains("rateLimited"))
        XCTAssertTrue(formatted.contains("Too many requests"))
        XCTAssertTrue(formatted.contains("retry_after"))
    }

    // MARK: - ToolChoice Tests

    func testToolChoiceCodable() throws {
        let choices: [ToolChoice] = [.auto, .none, .required, .specific("get_weather")]

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for choice in choices {
            let data = try encoder.encode(choice)
            let decoded = try decoder.decode(ToolChoice.self, from: data)
            XCTAssertEqual(decoded, choice)
        }
    }

    // MARK: - ConversationContext Tests

    func testConversationContextInitialization() throws {
        let tool = try Tool(
            name: "get_weather",
            description: "Get current weather",
            parameters: ToolParameters(properties: [:])
        )

        let context = ConversationContext(
            systemPrompt: "You are a helpful assistant",
            tools: [tool],
            initialMessage: "Hello"
        )

        XCTAssertEqual(context.messages.count, 1)
        XCTAssertEqual(context.messages[0].role, .user)
        XCTAssertEqual(context.messages[0].content, "Hello")
        XCTAssertFalse(context.requiresToolExecution)
    }

    func testConversationContextAddAssistantResponse() throws {
        let tool = try Tool(
            name: "get_weather",
            description: "Get current weather",
            parameters: ToolParameters(properties: [:])
        )

        var context = ConversationContext(
            systemPrompt: nil,
            tools: [tool],
            initialMessage: "What's the weather?"
        )

        let toolCall = ToolCall(id: "call_1", name: "get_weather", arguments: "{}")
        context.addAssistantResponse(text: nil, toolCalls: [toolCall])

        XCTAssertEqual(context.messages.count, 2)
        XCTAssertTrue(context.requiresToolExecution)
    }

    func testConversationContextAddToolResults() throws {
        let tool = try Tool(
            name: "get_weather",
            description: "Get current weather",
            parameters: ToolParameters(properties: [:])
        )

        var context = ConversationContext(
            systemPrompt: nil,
            tools: [tool],
            initialMessage: "What's the weather?"
        )

        let toolCall = ToolCall(id: "call_1", name: "get_weather", arguments: "{}")
        context.addAssistantResponse(text: nil, toolCalls: [toolCall])

        let result = ToolResult.error(id: "call_1", message: "Weather API unavailable")
        context.addToolResults([result])

        XCTAssertEqual(context.messages.count, 3)
        XCTAssertFalse(context.requiresToolExecution)
    }

    // MARK: - CompletionResponse Tests

    func testCompletionResponseWithToolCalls() {
        let toolCall = ToolCall(id: "call_1", name: "test", arguments: "{}")
        let response = CompletionResponse(
            text: "",
            model: "gpt-4",
            usage: TokenUsage(inputTokens: 10, outputTokens: 5),
            toolCalls: [toolCall]
        )

        XCTAssertTrue(response.requiresToolExecution)
        XCTAssertEqual(response.toolCalls.count, 1)
    }

    func testCompletionResponseWithoutToolCalls() {
        let response = CompletionResponse(
            text: "Hello",
            model: "gpt-4",
            usage: TokenUsage(inputTokens: 10, outputTokens: 5)
        )

        XCTAssertFalse(response.requiresToolExecution)
        XCTAssertTrue(response.toolCalls.isEmpty)
    }
}
