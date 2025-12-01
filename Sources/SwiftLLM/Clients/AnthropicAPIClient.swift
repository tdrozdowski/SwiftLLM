import Foundation

/// API client for Anthropic's Claude models
actor AnthropicAPIClient {
    private let apiKey: String
    private let baseURL: URL
    private let session: URLSession

    init(apiKey: String, baseURL: URL = URL(string: "https://api.anthropic.com")!) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.session = URLSession.shared
    }

    // MARK: - Messages API

    struct MessagesRequest: Codable {
        let model: String
        let messages: [Message]
        let system: String?
        let max_tokens: Int
        let temperature: Double?
        let top_p: Double?
        var stream: Bool?

        struct Message: Codable {
            let role: String
            let content: String
        }
    }

    struct MessagesResponse: Codable {
        let id: String
        let type: String
        let role: String
        let content: [Content]
        let model: String
        let stop_reason: String?
        let usage: Usage

        struct Content: Codable {
            let type: String
            let text: String?
        }

        struct Usage: Codable {
            let input_tokens: Int
            let output_tokens: Int
        }
    }

    func createMessage(request: MessagesRequest) async throws -> MessagesResponse {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("/v1/messages"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
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
            throw LLMError.networkError(NSError(domain: "AnthropicAPI", code: -1, userInfo: nil))
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
        return try decoder.decode(MessagesResponse.self, from: data)
    }

    nonisolated func streamMessage(request: MessagesRequest) -> AsyncThrowingStream<String, Error> {
        // Capture actor-isolated properties before the Task to avoid data races
        let capturedBaseURL = baseURL
        let capturedAPIKey = apiKey
        let capturedSession = session

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var urlRequest = URLRequest(url: capturedBaseURL.appendingPathComponent("/v1/messages"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue(capturedAPIKey, forHTTPHeaderField: "x-api-key")
                    urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    var streamRequest = request
                    streamRequest.stream = true

                    let encoder = JSONEncoder()
                    encoder.outputFormatting = .prettyPrinted
                    urlRequest.httpBody = try encoder.encode(streamRequest)

                    // Log request
                    SwiftLLMLogger.api.logRequest(
                        url: urlRequest.url?.absoluteString ?? "unknown",
                        method: "POST",
                        headers: urlRequest.allHTTPHeaderFields
                    )
                    if let body = urlRequest.httpBody, let bodyString = String(data: body, encoding: .utf8) {
                        SwiftLLMLogger.api.logRequestBody(bodyString)
                    }

                    let (bytes, response) = try await capturedSession.bytes(for: urlRequest)

                    // Log response
                    if let httpResponse = response as? HTTPURLResponse {
                        SwiftLLMLogger.api.logResponse(
                            statusCode: httpResponse.statusCode,
                            url: urlRequest.url?.absoluteString
                        )
                    }

                    guard let httpResponse = response as? HTTPURLResponse,
                          httpResponse.statusCode == 200 else {
                        throw LLMError.providerError("Stream request failed", code: nil)
                    }

                    for try await line in bytes.lines {
                        if line.hasPrefix("data: ") {
                            let jsonString = String(line.dropFirst(6))
                            if let data = jsonString.data(using: .utf8),
                               let event = try? JSONDecoder().decode(StreamEvent.self, from: data) {
                                if event.type == "content_block_delta",
                                   let delta = event.delta,
                                   let text = delta.text {
                                    continuation.yield(text)
                                }
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

    struct StreamEvent: Codable {
        let type: String
        let delta: Delta?

        struct Delta: Codable {
            let type: String?
            let text: String?
        }
    }

    // MARK: - Token Counting

    func estimateTokens(_ text: String) -> Int {
        // Rough approximation: ~4 characters per token
        // Anthropic doesn't provide a tokenizer API, so this is an estimate
        return max(1, text.count / 4)
    }
}
