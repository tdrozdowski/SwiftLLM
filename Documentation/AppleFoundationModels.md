# Apple Foundation Models (AFM) Guide

## Overview

Apple Foundation Models (AFM) provide on-device and server-based language models through the FoundationModels framework, announced at WWDC 2025. These models power Apple Intelligence and are available for third-party developers on macOS 26+ and iOS 26+.

## Model Variants

### On-Device AFM
- **Parameters**: 3 billion
- **Vision Transformer**: 300 million parameters
- **Architecture**: Transformer-based language model
- **Deployment**: Runs entirely on-device
- **Availability**: macOS 26+, iOS 26+ with Apple Silicon

### Server-Based AFM
- **Architecture**: Custom mixture-of-experts transformer
- **Vision Transformer**: 1 billion parameters
- **Parameters**: Undisclosed (larger than on-device)
- **Deployment**: Cloud-hosted on Apple servers
- **Availability**: macOS 26+, iOS 26+ (requires network)

## Capabilities Comparison

| Feature | On-Device | Server |
|---------|-----------|--------|
| **Context Window** | 8,192 tokens | 32,768 tokens |
| **Max Output** | 4,096 tokens | 4,096 tokens |
| **Privacy** | ✅ Complete (never leaves device) | ⚠️ Processed on Apple servers |
| **Network Required** | ❌ No | ✅ Yes |
| **Cost** | ✅ Free | ⚠️ Pricing TBD |
| **Latency** | ✅ Very low (~instant) | ⚠️ Network-dependent |
| **Processing Power** | 🔋 Uses device resources | ☁️ Uses Apple cloud |
| **Availability** | 📱 Offline capable | 🌐 Online only |
| **Model Quality** | Good for most tasks | Better for complex reasoning |

## When to Use On-Device AFM

### ✅ Ideal Use Cases

1. **Privacy-Critical Applications**
   - Personal journaling apps
   - Medical/health data processing
   - Financial analysis tools
   - Password/credential management
   - Personal note-taking and organization

2. **Offline-First Experiences**
   - Travel apps without connectivity
   - Field research tools
   - Airplane mode productivity
   - Remote location applications

3. **Real-Time, Low-Latency Tasks**
   - Live autocomplete/suggestions
   - Real-time translation
   - Voice assistant responses
   - Interactive games
   - Live coding assistants

4. **Cost-Sensitive Applications**
   - High-volume processing
   - Frequent small queries
   - Educational apps
   - Personal productivity tools

5. **Battery-Efficient Processing**
   - Mobile-first applications
   - Background processing
   - Always-on features

### ❌ Limitations

1. **Context Window**: 8K tokens (~6,000 words)
   - Not suitable for analyzing very long documents
   - Limited conversation history
   - Cannot process entire codebases

2. **Model Capabilities**: 3B parameters
   - Less capable than frontier models (GPT-5, Claude Opus)
   - May struggle with highly complex reasoning
   - Limited domain-specific knowledge

3. **Device Requirements**
   - Requires Apple Silicon (M-series or A-series chips)
   - macOS 26+ or iOS 26+
   - May not run on older devices

4. **Processing Power**
   - Uses device CPU/Neural Engine
   - Can impact battery life under heavy use
   - May throttle on resource-constrained devices

## When to Use Server-Based AFM

### ✅ Ideal Use Cases

1. **Complex Reasoning Tasks**
   - Advanced problem-solving
   - Multi-step analysis
   - Complex code generation
   - Scientific/technical writing

2. **Longer Context Requirements**
   - Processing longer documents (up to 32K tokens)
   - Extended conversation history
   - Large codebase analysis
   - Book/article summarization

3. **Resource-Constrained Devices**
   - Older devices without Apple Silicon
   - Battery-saving mode
   - Devices under heavy load
   - When device is throttling

4. **Quality Over Speed**
   - When accuracy is paramount
   - Complex creative writing
   - Professional content generation
   - Detailed research tasks

### ❌ Limitations

1. **Privacy Concerns**
   - Data sent to Apple servers
   - Not suitable for highly sensitive information
   - Requires trust in Apple's infrastructure

2. **Network Dependency**
   - Requires internet connection
   - Subject to network latency
   - May fail in poor connectivity
   - Cannot work offline

3. **Rate Limiting**
   - May encounter `GenerationError.rateLimited` errors
   - Background tasks more likely to be throttled
   - Unknown usage quotas/limits

4. **Cost** (TBD)
   - Pricing not yet announced
   - May incur costs for heavy usage
   - Could impact app economics

5. **Availability**
   - Depends on Apple server availability
   - May have regional restrictions
   - Subject to Apple service status

## Performance Characteristics

### On-Device AFM

**Strengths**:
- Instant response time (no network latency)
- Consistent performance regardless of internet
- Privacy-preserving by design
- No per-request costs
- Works anywhere, anytime

**Weaknesses**:
- Limited by device hardware
- Smaller context window
- Less sophisticated reasoning
- May drain battery under sustained load

### Server-Based AFM

**Strengths**:
- More powerful model (mixture-of-experts)
- Larger context window (32K tokens)
- Better at complex tasks
- Doesn't drain device battery
- Consistent quality across devices

**Weaknesses**:
- Network latency adds delay
- Requires stable internet connection
- Privacy trade-offs
- Potential costs
- Rate limiting

## Best Practices

### Choosing Between On-Device and Server

```swift
import SwiftLLM

// Privacy-first: Use on-device for sensitive data
let privateProvider = AppleIntelligenceProvider.onDevice(
    instructions: "You are a personal finance assistant."
)

// Quality-first: Use server for complex tasks
let complexProvider = AppleIntelligenceProvider.server(
    instructions: "You are an expert software architect."
)
```

### Hybrid Approach

Consider using both models strategically:

```swift
// Fast, private autocomplete with on-device
let quickResponses = AppleIntelligenceProvider.onDevice()

// Deep analysis with server model
let deepAnalysis = AppleIntelligenceProvider.server()

// Example: Use on-device for drafting, server for refinement
let draft = try await quickResponses.generateCompletion(
    prompt: "Draft a brief email about...",
    systemPrompt: nil,
    options: GenerationOptions(temperature: 0.7)
)

let refined = try await deepAnalysis.generateCompletion(
    prompt: "Refine this email for professionalism: \(draft.text)",
    systemPrompt: "You are a professional communication expert.",
    options: GenerationOptions(temperature: 0.3)
)
```

### Rate Limiting Guidance

Apple's documentation warns about rate limiting, especially for background tasks:

> **Important**: If running in the background, use the non-streaming `respond(to:options:)` method to reduce the likelihood of encountering `LanguageModelSession.GenerationError.rateLimited(_:)` errors.

```swift
// Prefer non-streaming for background tasks
Task.detached {
    let response = try await session.respond(to: prompt, options: options)
    // Process response
}

// Use streaming only for foreground, user-facing interactions
let stream = session.streamResponse(to: prompt, options: options)
for try await chunk in stream {
    // Update UI in real-time
}
```

## Comparison with Other Providers

| Feature | AFM On-Device | AFM Server | Claude Opus 4.5 | GPT-5.1 | Grok 4.1 |
|---------|---------------|------------|-----------------|---------|----------|
| **Privacy** | ✅ Complete | ⚠️ Cloud | ⚠️ Cloud | ⚠️ Cloud | ⚠️ Cloud |
| **Offline** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No |
| **Cost** | ✅ Free | TBD | $5-25/1M | $5-15/1M | $0.20-10/1M |
| **Context** | 8K | 32K | 200K | 200K-1M | 500K-2M |
| **Quality** | Good | Better | Excellent | Excellent | Excellent |
| **Speed** | ⚡ Instant | 🌐 Fast | 🌐 Fast | 🌐 Fast | 🌐 Fast |

## Technical Details

### Framework Requirements
- **macOS**: 26.0+
- **iOS**: 26.0+
- **iPadOS**: 26.0+
- **visionOS**: 26.0+
- **Mac Catalyst**: 26.0+

### Hardware Requirements (On-Device)
- **Mac**: Apple Silicon (M1 or later)
- **iPhone**: A17 Pro or later
- **iPad**: M1 or later

### API Capabilities
- ✅ Streaming responses via `ResponseStream`
- ✅ Structured output via `Generable` protocol
- ✅ Tool calling with constrained generation
- ✅ LoRA adapter fine-tuning
- ✅ Safety guardrails built-in
- ✅ Vision capabilities (image understanding)
- ✅ Custom generation options (temperature, max tokens, sampling)

## Example Use Cases by Category

### Education
- **On-Device**: Quiz generation from notes, flashcard creation, quick explanations
- **Server**: Essay grading, complex problem solving, detailed research assistance

### Productivity
- **On-Device**: Email drafting, meeting notes, quick summaries
- **Server**: Strategic planning, complex document analysis, presentation creation

### Creative Writing
- **On-Device**: Brainstorming, quick edits, character names
- **Server**: Story development, plot analysis, world-building

### Development
- **On-Device**: Code completion, quick bug fixes, inline documentation
- **Server**: Architecture review, complex refactoring, algorithm design

### Business
- **On-Device**: Quick customer responses, internal notes, data entry assistance
- **Server**: Market analysis, strategic reports, competitive intelligence

## Future Considerations

As of November 2025, Apple has not announced:
- Pricing for server-based AFM
- Exact rate limits or quotas
- Regional availability details
- Fine-tuning capabilities for third-party developers
- Model versioning/update schedule

Monitor Apple's developer documentation for updates.

## References

- [Foundation Models Framework - Apple Developer](https://developer.apple.com/documentation/FoundationModels)
- [Apple Intelligence Foundation Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- [Generating Content with Foundation Models](https://developer.apple.com/documentation/foundationmodels/generating-content-and-performing-tasks-with-foundation-models)
