import Foundation

/// API client for OpenAI's GPT models
actor OpenAIAPIClient {
    private let apiKey: String
    private let baseURL: URL
    private let session: URLSession

    init(apiKey: String, baseURL: URL = URL(string: "https://api.openai.com")!) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.session = URLSession.shared
    }

    // MARK: - Chat Completions API

    struct ChatCompletionRequest: Codable, Sendable {
        let model: String
        let messages: [Message]
        let temperature: Double?
        let max_tokens: Int?
        let top_p: Double?
        let frequency_penalty: Double?
        let presence_penalty: Double?
        let stop: [String]?
        var stream: Bool?
        let response_format: ResponseFormat?
        let tools: [OpenAITool]?
        let tool_choice: OpenAIToolChoice?

        init(
            model: String,
            messages: [Message],
            temperature: Double? = nil,
            max_tokens: Int? = nil,
            top_p: Double? = nil,
            frequency_penalty: Double? = nil,
            presence_penalty: Double? = nil,
            stop: [String]? = nil,
            stream: Bool? = nil,
            response_format: ResponseFormat? = nil,
            tools: [OpenAITool]? = nil,
            tool_choice: OpenAIToolChoice? = nil
        ) {
            self.model = model
            self.messages = messages
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.top_p = top_p
            self.frequency_penalty = frequency_penalty
            self.presence_penalty = presence_penalty
            self.stop = stop
            self.stream = stream
            self.response_format = response_format
            self.tools = tools
            self.tool_choice = tool_choice
        }

        struct Message: Codable, Sendable {
            let role: String
            let content: String?
            let tool_calls: [OpenAIToolCall]?
            let tool_call_id: String?

            init(role: String, content: String?, tool_calls: [OpenAIToolCall]? = nil, tool_call_id: String? = nil) {
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
                self.tool_call_id = tool_call_id
            }
        }

        struct ResponseFormat: Codable, Sendable {
            let type: String // "text" or "json_object"
        }

        struct OpenAITool: Codable, Sendable {
            let type: String // Always "function"
            let function: Function

            struct Function: Codable, Sendable {
                let name: String
                let description: String
                let parameters: AnyCodable
            }
        }

        struct OpenAIToolCall: Codable, Sendable {
            let id: String
            let type: String // Always "function"
            let function: FunctionCall

            struct FunctionCall: Codable, Sendable {
                let name: String
                let arguments: String
            }
        }
    }

    // Type-safe Sendable wrapper for JSON values
    struct AnyCodable: Codable, Sendable {
        let value: SendableValue

        init(_ value: SendableValue) {
            self.value = value
        }

        enum SendableValue: Sendable {
            case string(String)
            case int(Int)
            case double(Double)
            case bool(Bool)
            case array([SendableValue])
            case dictionary([String: SendableValue])
            case null

            init(_ any: Any) throws {
                if let string = any as? String {
                    self = .string(string)
                } else if let int = any as? Int {
                    self = .int(int)
                } else if let double = any as? Double {
                    self = .double(double)
                } else if let bool = any as? Bool {
                    self = .bool(bool)
                } else if let array = any as? [Any] {
                    self = .array(try array.map { try SendableValue($0) })
                } else if let dict = any as? [String: Any] {
                    self = .dictionary(try dict.mapValues { try SendableValue($0) })
                } else if any is NSNull {
                    self = .null
                } else {
                    throw EncodingError.invalidValue(any, EncodingError.Context(codingPath: [], debugDescription: "Unsupported type"))
                }
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch value {
            case .string(let string):
                try container.encode(string)
            case .int(let int):
                try container.encode(int)
            case .double(let double):
                try container.encode(double)
            case .bool(let bool):
                try container.encode(bool)
            case .array(let array):
                try container.encode(array.map { AnyCodable($0) })
            case .dictionary(let dict):
                try container.encode(dict.mapValues { AnyCodable($0) })
            case .null:
                try container.encodeNil()
            }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let string = try? container.decode(String.self) {
                value = .string(string)
            } else if let int = try? container.decode(Int.self) {
                value = .int(int)
            } else if let double = try? container.decode(Double.self) {
                value = .double(double)
            } else if let bool = try? container.decode(Bool.self) {
                value = .bool(bool)
            } else if let array = try? container.decode([AnyCodable].self) {
                value = .array(array.map { $0.value })
            } else if let dict = try? container.decode([String: AnyCodable].self) {
                value = .dictionary(dict.mapValues { $0.value })
            } else if container.decodeNil() {
                value = .null
            } else {
                throw DecodingError.dataCorruptedError(in: container, debugDescription: "Cannot decode value")
            }
        }
    }

    struct ChatCompletionResponse: Codable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage

        struct Choice: Codable {
            let index: Int
            let message: Message
            let finish_reason: String?

            struct Message: Codable {
                let role: String
                let content: String?
                let tool_calls: [ChatCompletionRequest.OpenAIToolCall]?
            }
        }

        struct Usage: Codable {
            let prompt_tokens: Int
            let completion_tokens: Int
            let total_tokens: Int
        }
    }

    func createChatCompletion(request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("/v1/chat/completions"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        urlRequest.httpBody = try encoder.encode(request)

        // Log request
        SwiftLLMLogger.api.logRequest(
            url: urlRequest.url?.absoluteString ?? "unknown",
            method: "POST",
            headers: urlRequest.allHTTPHeaderFields
        )
        if let body = urlRequest.httpBody, let bodyString = String(data: body, encoding: .utf8) {
            SwiftLLMLogger.api.logRequestBody(bodyString)
        }

        let (data, response) = try await session.data(for: urlRequest)

        // Log response
        if let httpResponse = response as? HTTPURLResponse {
            SwiftLLMLogger.api.logResponse(
                statusCode: httpResponse.statusCode,
                url: urlRequest.url?.absoluteString
            )
        }
        if let responseString = String(data: data, encoding: .utf8) {
            SwiftLLMLogger.api.logResponseBody(responseString)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMError.networkError(NSError(domain: "OpenAIAPI", code: -1, userInfo: nil))
        }

        guard httpResponse.statusCode == 200 else {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            switch httpResponse.statusCode {
            case 401:
                throw LLMError.authenticationFailed(errorMessage)
            case 429:
                let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After")
                    .flatMap { TimeInterval($0) }
                throw LLMError.rateLimitExceeded(retryAfter: retryAfter)
            case 400:
                throw LLMError.invalidRequest(errorMessage)
            default:
                throw LLMError.providerError(errorMessage, code: "\(httpResponse.statusCode)")
            }
        }

        let decoder = JSONDecoder()
        return try decoder.decode(ChatCompletionResponse.self, from: data)
    }

    nonisolated func streamChatCompletion(request: ChatCompletionRequest) -> AsyncThrowingStream<String, Error> {
        // Capture actor-isolated properties before the Task to avoid data races
        let capturedBaseURL = baseURL
        let capturedAPIKey = apiKey
        let capturedSession = session

        return AsyncThrowingStream { continuation in
            Task { @Sendable in
                do {
                    var urlRequest = URLRequest(url: capturedBaseURL.appendingPathComponent("/v1/chat/completions"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("Bearer \(capturedAPIKey)", forHTTPHeaderField: "Authorization")
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    var streamRequest = request
                    streamRequest.stream = true

                    let encoder = JSONEncoder()
                    urlRequest.httpBody = try encoder.encode(streamRequest)

                    let (bytes, response) = try await capturedSession.bytes(for: urlRequest)

                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw LLMError.networkError(NSError(domain: "OpenAIAPI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response type"]))
                    }

                    guard httpResponse.statusCode == 200 else {
                        // Try to read error body for better error messages
                        var errorMessage = "Stream request failed with status \(httpResponse.statusCode)"
                        var errorData = Data()
                        for try await byte in bytes {
                            errorData.append(byte)
                            // Limit error body size to prevent memory issues
                            if errorData.count > 4096 { break }
                        }
                        if let errorBody = String(data: errorData, encoding: .utf8), !errorBody.isEmpty {
                            errorMessage += ": \(errorBody)"
                        }

                        switch httpResponse.statusCode {
                        case 401:
                            throw LLMError.authenticationFailed(errorMessage)
                        case 429:
                            let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After")
                                .flatMap { TimeInterval($0) }
                            throw LLMError.rateLimitExceeded(retryAfter: retryAfter)
                        case 400:
                            throw LLMError.invalidRequest(errorMessage)
                        default:
                            throw LLMError.providerError(errorMessage, code: "\(httpResponse.statusCode)")
                        }
                    }

                    for try await line in bytes.lines {
                        if line.hasPrefix("data: ") {
                            let jsonString = String(line.dropFirst(6))
                            if jsonString == "[DONE]" {
                                break
                            }
                            if let data = jsonString.data(using: .utf8),
                               let chunk = try? JSONDecoder().decode(StreamChunk.self, from: data),
                               let delta = chunk.choices.first?.delta.content {
                                continuation.yield(delta)
                            }
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    struct StreamChunk: Codable {
        let choices: [Choice]

        struct Choice: Codable {
            let delta: Delta

            struct Delta: Codable {
                let content: String?
            }
        }
    }

    // MARK: - Token Counting

    func estimateTokens(_ text: String) -> Int {
        // Rough approximation: ~4 characters per token for GPT models
        // OpenAI provides tiktoken library but not via API
        return max(1, text.count / 4)
    }
}
