# SwiftLLM Logging

SwiftLLM uses Apple's unified logging system (`os.log`) for trace logging and troubleshooting.

## Log Categories

- **API**: HTTP requests and responses from all API clients
- **Provider**: Provider-level operations (tool calling, completions)
- **Tools**: Tool execution and function calls
- **General**: General library operations
- **Error**: Errors and warnings

## Viewing Logs

### Option 1: Console.app (easiest)
1. Open Console.app (`/Applications/Utilities/Console.app`)
2. Filter by subsystem: `subsystem:com.swiftllm`
3. Run your code - logs appear in real-time

### Option 2: Terminal with log command

Stream all SwiftLLM logs:
```bash
log stream --predicate 'subsystem == "com.swiftllm"' --level debug
```

Filter by category:
```bash
# API requests/responses only
log stream --predicate 'subsystem == "com.swiftllm" AND category == "API"' --level debug

# Tool calling only
log stream --predicate 'subsystem == "com.swiftllm" AND category == "Tools"' --level debug

# Errors only
log stream --predicate 'subsystem == "com.swiftllm" AND category == "Error"'
```

Show recent logs:
```bash
# Last 5 minutes
log show --predicate 'subsystem == "com.swiftllm"' --last 5m --debug

# Last hour
log show --predicate 'subsystem == "com.swiftllm"' --last 1h --debug
```

### Option 3: Xcode Console
When running in Xcode, logs automatically appear in the debug console.

## Log Levels

- **Info**: High-level operations (requests sent, responses received)
- **Debug**: Detailed information (headers, bodies, arguments)
- **Error**: Errors and failures

## Example Output

```
→ POST https://api.x.ai/v1/chat/completions
  Headers: ["Content-Type": "application/json", "Authorization": "Bearer xai-..."]
  Request Body: {
    "model": "grok-2-latest",
    "messages": [{"role": "user", "content": "Hello"}],
    "response_format": {"type": "json_object"}
  }
← Response 200 https://api.x.ai/v1/chat/completions
  Response Body: {"id": "chatcmpl-123", "choices": [...]}
```

## Troubleshooting

If logs aren't appearing:

1. **Check log level**: Use `--level debug` to see debug-level logs
2. **Check timing**: Start `log stream` **before** running your code
3. **Check Console.app**: Sometimes easier than command-line
4. **Check filter**: Make sure subsystem is `com.swiftllm` (not a typo)

## Performance

The unified logging system is highly optimized:
- Logs are asynchronous and don't block your code
- Debug logs are stripped in release builds
- No performance impact when not actively streaming logs
