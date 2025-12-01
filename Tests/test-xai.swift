#!/usr/bin/env swift

import Foundation

// This script tests the xAI implementation
// To run: swift Tests/test-xai.swift

// MARK: - Load API Key from .env

func loadEnvFile() -> [String: String] {
    let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(".env")

    guard let contents = try? String(contentsOf: fileURL, encoding: .utf8) else {
        print("❌ Could not read .env file")
        return [:]
    }

    var env: [String: String] = [:]
    for line in contents.components(separatedBy: .newlines) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }

        let parts = trimmed.components(separatedBy: "=")
        guard parts.count == 2 else { continue }

        let key = parts[0].trimmingCharacters(in: .whitespaces)
        let value = parts[1].trimmingCharacters(in: .whitespaces)
        env[key] = value
    }

    return env
}

let env = loadEnvFile()
guard let apiKey = env["XAI_API_KEY"], apiKey != "your-key-here" else {
    print("❌ Please set XAI_API_KEY in .env file")
    exit(1)
}

print("✅ API key loaded from .env")
print("🧪 Testing xAI provider...")

// MARK: - Simple Test

@main
struct TestRunner {
    static func main() async {
        do {
            // Import SwiftLLM here when running as a proper test
            print("⚠️  This is a template test script")
            print("   To test properly, create a Swift Package executable or use:")
            print("   swift run YourTestExecutable")
            print("")
            print("✅ API key is loaded and ready")
            print("   Key prefix: \(String(apiKey.prefix(10)))...")

        } catch {
            print("❌ Test failed: \(error)")
        }
    }
}
