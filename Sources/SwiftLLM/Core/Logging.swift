import Foundation
import os.log

/// Centralized logging for SwiftLLM
public enum SwiftLLMLogger {
    private static let subsystem = "com.swiftllm"

    /// Logger for API client operations (requests, responses)
    public static let api = Logger(subsystem: subsystem, category: "API")

    /// Logger for provider operations
    public static let provider = Logger(subsystem: subsystem, category: "Provider")

    /// Logger for tool calling and function execution
    public static let tools = Logger(subsystem: subsystem, category: "Tools")

    /// Logger for general library operations
    public static let general = Logger(subsystem: subsystem, category: "General")

    /// Logger for errors and warnings
    public static let error = Logger(subsystem: subsystem, category: "Error")
}

/// Extension to make logging more ergonomic
extension Logger {
    /// Log a network request
    func logRequest(url: String, method: String = "POST", headers: [String: String]? = nil) {
        self.info("→ \(method, privacy: .public) \(url, privacy: .public)")
        if let headers = headers {
            // Redact Authorization header but show others
            var safeHeaders = headers
            if safeHeaders["Authorization"] != nil {
                safeHeaders["Authorization"] = "Bearer <redacted>"
            }
            self.debug("  Headers: \(String(describing: safeHeaders), privacy: .public)")
        }
    }

    /// Log a request body
    func logRequestBody(_ body: String) {
        self.debug("  Request Body: \(body, privacy: .public)")
    }

    /// Log a network response
    func logResponse(statusCode: Int, url: String? = nil) {
        if statusCode >= 200 && statusCode < 300 {
            self.info("← Response \(statusCode, privacy: .public) \(url ?? "", privacy: .public)")
        } else {
            self.error("← Response \(statusCode, privacy: .public) \(url ?? "", privacy: .public)")
        }
    }

    /// Log a response body
    func logResponseBody(_ body: String, maxLength: Int = 1000) {
        let truncated = body.count > maxLength ? String(body.prefix(maxLength)) + "... (truncated)" : body
        self.debug("  Response Body: \(truncated, privacy: .public)")
    }

    /// Log an error with context
    func logError(_ error: Error, context: String) {
        self.error("❌ \(context): \(error.localizedDescription)")
    }

    /// Log a tool call
    func logToolCall(name: String, id: String) {
        self.info("🔧 Tool call: \(name) (id: \(id))")
    }

    /// Log tool arguments
    func logToolArguments(_ arguments: String) {
        self.debug("  Arguments: \(arguments)")
    }

    /// Log tool result
    func logToolResult(success: Bool, id: String) {
        if success {
            self.info("✅ Tool result: \(id)")
        } else {
            self.error("❌ Tool error: \(id)")
        }
    }
}
