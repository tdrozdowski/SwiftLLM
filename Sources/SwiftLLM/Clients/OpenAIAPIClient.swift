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
        let tool_choice: ToolChoiceValue?

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
            tool_choice: ToolChoiceValue? = nil
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

        // Type-erased Codable wrapper for tool_choice
        enum ToolChoiceValue: Codable, Sendable {
            case string(String)
            case object([String: AnyCodable])

            func encode(to encoder: Encoder) throws {
                var container = encoder.singleValueContainer()
                switch self {
                case .string(let value):
                    try container.encode(value)
                case .object(let dict):
                    try container.encode(dict)
                }
            }

            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let string = try? container.decode(String.self) {
                    self = .string(string)
                } else if let dict = try? container.decode([String: AnyCodable].self) {
                    self = .object(dict)
                } else {
                    throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid tool_choice")
                }
            }
        }
    }

    // Type-erased Codable wrapper
    struct AnyCodable: Codable, @unchecked Sendable {
        let value: Any

        init(_ value: Any) {
            self.value = value
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            if let string = value as? String {
                try container.encode(string)
            } else if let int = value as? Int {
                try container.encode(int)
            } else if let double = value as? Double {
                try container.encode(double)
            } else if let bool = value as? Bool {
                try container.encode(bool)
            } else if let array = value as? [Any] {
                try container.encode(array.map { AnyCodable($0) })
            } else if let dict = value as? [String: Any] {
                try container.encode(dict.mapValues { AnyCodable($0) })
            } else if value is NSNull {
                try container.encodeNil()
            } else {
                throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: encoder.codingPath, debugDescription: "Invalid type"))
            }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let string = try? container.decode(String.self) {
                value = string
            } else if let int = try? container.decode(Int.self) {
                value = int
            } else if let double = try? container.decode(Double.self) {
                value = double
            } else if let bool = try? container.decode(Bool.self) {
                value = bool
            } else if let array = try? container.decode([AnyCodable].self) {
                value = array.map { $0.value }
            } else if let dict = try? container.decode([String: AnyCodable].self) {
                value = dict.mapValues { $0.value }
            } else if container.decodeNil() {
                value = NSNull()
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
        urlRequest.httpBody = try encoder.encode(request)

        let (data, response) = try await session.data(for: urlRequest)

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
        AsyncThrowingStream { continuation in
            Task { @Sendable in
                do {
                    var urlRequest = URLRequest(url: baseURL.appendingPathComponent("/v1/chat/completions"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    var streamRequest = request
                    streamRequest.stream = true

                    let encoder = JSONEncoder()
                    urlRequest.httpBody = try encoder.encode(streamRequest)

                    let (bytes, response) = try await session.bytes(for: urlRequest)

                    guard let httpResponse = response as? HTTPURLResponse,
                          httpResponse.statusCode == 200 else {
                        throw LLMError.providerError("Stream request failed", code: nil)
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
