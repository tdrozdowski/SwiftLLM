import Foundation

/// API client for xAI's Grok models
/// Note: xAI uses OpenAI-compatible API
actor XAIAPIClient {
    private let apiKey: String
    private let baseURL: URL
    private let session: URLSession

    init(apiKey: String, baseURL: URL = URL(string: "https://api.x.ai")!) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.session = URLSession.shared
    }

    // MARK: - Chat Completions API (OpenAI-compatible)

    struct ChatCompletionRequest: Codable {
        let model: String
        let messages: [Message]
        let temperature: Double?
        let max_tokens: Int?
        let top_p: Double?
        var stream: Bool?

        struct Message: Codable {
            let role: String
            let content: String
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
            throw LLMError.networkError(NSError(domain: "xAIAPI", code: -1, userInfo: nil))
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
            Task {
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
        // Rough approximation
        return max(1, text.count / 4)
    }
}
