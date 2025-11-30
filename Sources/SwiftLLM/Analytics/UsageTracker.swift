import Foundation

/// Tracks token usage and costs across multiple LLM requests
public actor UsageTracker {
    /// Individual usage record
    public struct UsageRecord: Sendable, Codable {
        public let timestamp: Date
        public let providerId: String
        public let model: String
        public let usage: TokenUsage
        public let cost: Double?
        public let metadata: [String: String]?

        public init(
            timestamp: Date = Date(),
            providerId: String,
            model: String,
            usage: TokenUsage,
            cost: Double? = nil,
            metadata: [String: String]? = nil
        ) {
            self.timestamp = timestamp
            self.providerId = providerId
            self.model = model
            self.usage = usage
            self.cost = cost
            self.metadata = metadata
        }
    }

    /// Aggregated usage statistics
    public struct UsageStats: Sendable {
        public let totalInputTokens: Int
        public let totalOutputTokens: Int
        public let totalTokens: Int
        public let totalCost: Double
        public let requestCount: Int
        public let averageTokensPerRequest: Double
        public let byProvider: [String: ProviderStats]
        public let byModel: [String: ModelStats]

        public struct ProviderStats: Sendable {
            public let inputTokens: Int
            public let outputTokens: Int
            public let cost: Double
            public let requestCount: Int
        }

        public struct ModelStats: Sendable {
            public let inputTokens: Int
            public let outputTokens: Int
            public let cost: Double
            public let requestCount: Int
        }
    }

    private var records: [UsageRecord] = []

    public init() {}

    /// Record usage from a completion response
    public func record(
        response: CompletionResponse,
        provider: any LLMProvider
    ) {
        let cost = calculateCost(usage: response.usage, pricing: provider.capabilities.pricing)
        let record = UsageRecord(
            providerId: provider.id,
            model: response.model,
            usage: response.usage,
            cost: cost,
            metadata: response.metadata
        )
        records.append(record)
    }

    /// Record usage manually
    public func record(_ record: UsageRecord) {
        records.append(record)
    }

    /// Get all usage records
    public func allRecords() -> [UsageRecord] {
        records
    }

    /// Get records filtered by date range
    public func records(from startDate: Date, to endDate: Date) -> [UsageRecord] {
        records.filter { $0.timestamp >= startDate && $0.timestamp <= endDate }
    }

    /// Get records for a specific provider
    public func records(forProvider providerId: String) -> [UsageRecord] {
        records.filter { $0.providerId == providerId }
    }

    /// Get records for a specific model
    public func records(forModel model: String) -> [UsageRecord] {
        records.filter { $0.model == model }
    }

    /// Get aggregated statistics
    public func stats() -> UsageStats {
        var totalInput = 0
        var totalOutput = 0
        var totalCost = 0.0
        var byProvider: [String: (input: Int, output: Int, cost: Double, count: Int)] = [:]
        var byModel: [String: (input: Int, output: Int, cost: Double, count: Int)] = [:]

        for record in records {
            totalInput += record.usage.inputTokens
            totalOutput += record.usage.outputTokens
            totalCost += record.cost ?? 0

            // Aggregate by provider
            var providerStats = byProvider[record.providerId] ?? (0, 0, 0, 0)
            providerStats.input += record.usage.inputTokens
            providerStats.output += record.usage.outputTokens
            providerStats.cost += record.cost ?? 0
            providerStats.count += 1
            byProvider[record.providerId] = providerStats

            // Aggregate by model
            var modelStats = byModel[record.model] ?? (0, 0, 0, 0)
            modelStats.input += record.usage.inputTokens
            modelStats.output += record.usage.outputTokens
            modelStats.cost += record.cost ?? 0
            modelStats.count += 1
            byModel[record.model] = modelStats
        }

        let totalTokens = totalInput + totalOutput
        let requestCount = records.count

        return UsageStats(
            totalInputTokens: totalInput,
            totalOutputTokens: totalOutput,
            totalTokens: totalTokens,
            totalCost: totalCost,
            requestCount: requestCount,
            averageTokensPerRequest: requestCount > 0 ? Double(totalTokens) / Double(requestCount) : 0,
            byProvider: byProvider.mapValues {
                UsageStats.ProviderStats(inputTokens: $0.input, outputTokens: $0.output, cost: $0.cost, requestCount: $0.count)
            },
            byModel: byModel.mapValues {
                UsageStats.ModelStats(inputTokens: $0.input, outputTokens: $0.output, cost: $0.cost, requestCount: $0.count)
            }
        )
    }

    /// Get stats for a date range
    public func stats(from startDate: Date, to endDate: Date) -> UsageStats {
        let filteredRecords = records(from: startDate, to: endDate)
        var totalInput = 0
        var totalOutput = 0
        var totalCost = 0.0
        var byProvider: [String: (input: Int, output: Int, cost: Double, count: Int)] = [:]
        var byModel: [String: (input: Int, output: Int, cost: Double, count: Int)] = [:]

        for record in filteredRecords {
            totalInput += record.usage.inputTokens
            totalOutput += record.usage.outputTokens
            totalCost += record.cost ?? 0

            var providerStats = byProvider[record.providerId] ?? (0, 0, 0, 0)
            providerStats.input += record.usage.inputTokens
            providerStats.output += record.usage.outputTokens
            providerStats.cost += record.cost ?? 0
            providerStats.count += 1
            byProvider[record.providerId] = providerStats

            var modelStats = byModel[record.model] ?? (0, 0, 0, 0)
            modelStats.input += record.usage.inputTokens
            modelStats.output += record.usage.outputTokens
            modelStats.cost += record.cost ?? 0
            modelStats.count += 1
            byModel[record.model] = modelStats
        }

        let totalTokens = totalInput + totalOutput
        let requestCount = filteredRecords.count

        return UsageStats(
            totalInputTokens: totalInput,
            totalOutputTokens: totalOutput,
            totalTokens: totalTokens,
            totalCost: totalCost,
            requestCount: requestCount,
            averageTokensPerRequest: requestCount > 0 ? Double(totalTokens) / Double(requestCount) : 0,
            byProvider: byProvider.mapValues {
                UsageStats.ProviderStats(inputTokens: $0.input, outputTokens: $0.output, cost: $0.cost, requestCount: $0.count)
            },
            byModel: byModel.mapValues {
                UsageStats.ModelStats(inputTokens: $0.input, outputTokens: $0.output, cost: $0.cost, requestCount: $0.count)
            }
        )
    }

    /// Clear all records
    public func reset() {
        records.removeAll()
    }

    /// Export records as JSON data
    public func exportJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(records)
    }

    /// Import records from JSON data
    public func importJSON(_ data: Data) throws {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let importedRecords = try decoder.decode([UsageRecord].self, from: data)
        records.append(contentsOf: importedRecords)
    }

    // MARK: - Private

    private func calculateCost(usage: TokenUsage, pricing: LLMPricing?) -> Double? {
        guard let pricing = pricing else { return nil }
        let inputCost = Double(usage.inputTokens) * pricing.inputCostPer1M / 1_000_000
        let outputCost = Double(usage.outputTokens) * pricing.outputCostPer1M / 1_000_000
        return inputCost + outputCost
    }
}

// MARK: - Convenience Extensions

extension UsageTracker.UsageStats: CustomStringConvertible {
    public var description: String {
        """
        Usage Statistics:
          Total Requests: \(requestCount)
          Total Tokens: \(totalTokens) (input: \(totalInputTokens), output: \(totalOutputTokens))
          Average Tokens/Request: \(String(format: "%.1f", averageTokensPerRequest))
          Total Cost: $\(String(format: "%.4f", totalCost))
        """
    }
}

extension TokenUsage {
    /// Calculate cost for this usage given pricing
    public func cost(with pricing: LLMPricing) -> Double {
        let inputCost = Double(inputTokens) * pricing.inputCostPer1M / 1_000_000
        let outputCost = Double(outputTokens) * pricing.outputCostPer1M / 1_000_000
        return inputCost + outputCost
    }
}
