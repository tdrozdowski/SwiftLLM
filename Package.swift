// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftLLM",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "SwiftLLM",
            targets: ["SwiftLLM"]
        ),
        .executable(
            name: "xai-test",
            targets: ["XAIManualTest"]
        )
    ],
    targets: [
        .target(
            name: "SwiftLLM",
            dependencies: []
        ),
        .executableTarget(
            name: "XAIManualTest",
            dependencies: ["SwiftLLM"],
            path: "Tests/XAIManualTest"
        ),
        .testTarget(
            name: "SwiftLLMTests",
            dependencies: ["SwiftLLM"]
        ),
        .testTarget(
            name: "SwiftLLMBenchmarks",
            dependencies: ["SwiftLLM"]
        )
    ]
)
