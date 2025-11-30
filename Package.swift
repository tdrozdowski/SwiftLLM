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
        )
    ],
    targets: [
        .target(
            name: "SwiftLLM",
            dependencies: []
        ),
        .testTarget(
            name: "SwiftLLMTests",
            dependencies: ["SwiftLLM"]
        )
    ]
)
