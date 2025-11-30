import Foundation

/// Errors that can occur during LLM operations
public enum LLMError: Error, LocalizedError, Sendable {
    /// API authentication failed
    case authenticationFailed(String)

    /// Rate limit exceeded
    case rateLimitExceeded(retryAfter: TimeInterval?)

    /// Invalid request (bad parameters, etc.)
    case invalidRequest(String)

    /// Provider-specific error
    case providerError(String, code: String?)

    /// Network error
    case networkError(Error)

    /// Parsing/decoding error
    case decodingError(String)

    /// Context length exceeded
    case contextLengthExceeded(requested: Int, maximum: Int)

    /// Unsupported feature for this provider
    case unsupportedFeature(String)

    /// Unknown error
    case unknown(Error)

    public var errorDescription: String? {
        switch self {
        case .authenticationFailed(let message):
            return "Authentication failed: \(message)"
        case .rateLimitExceeded(let retryAfter):
            if let retry = retryAfter {
                return "Rate limit exceeded. Retry after \(retry) seconds."
            }
            return "Rate limit exceeded."
        case .invalidRequest(let message):
            return "Invalid request: \(message)"
        case .providerError(let message, let code):
            if let code = code {
                return "Provider error (\(code)): \(message)"
            }
            return "Provider error: \(message)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .decodingError(let message):
            return "Decoding error: \(message)"
        case .contextLengthExceeded(let requested, let maximum):
            return "Context length exceeded: requested \(requested), maximum \(maximum)"
        case .unsupportedFeature(let feature):
            return "Unsupported feature: \(feature)"
        case .unknown(let error):
            return "Unknown error: \(error.localizedDescription)"
        }
    }
}
