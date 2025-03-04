# Agio: A Rust Client for OpenAI Agents

A Rust library for building agent-based systems with OpenAI's API.

## Overview

Agio provides a structured interface for interacting with OpenAI's API, with a focus on tool calling capabilities and agent-based workflows. It handles conversation state management, token counting, tool execution, and now includes WebSocket support for OpenAI's "Realtime" Beta API.

## Key Components

- **Agent**: Main entry point for conversational interactions
- **OpenAIClient**: Handles HTTP API communication
- **WebSocketClient**: Supports OpenAI's "Realtime" Beta API
- **ToolRegistry**: Manages the tools available to the agent
- **Config**: Provides configuration options for API requests

## Dependencies

This library has the following key dependencies:
- Rust 2024 edition
- tokio async runtime
- serde/serde_json for serialization
- reqwest for HTTP communication
- tokio-tungstenite for WebSocket support
- tiktoken-rs for token counting
- thiserror for error handling

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
agio = "0.1.0"
```

You'll need to provide your OpenAI API key to use this library.

## Basic Usage

```rust
use agio::{Agent, AgentBuilder, Config};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").expect("Missing API key");
    
    // Create agent with configuration
    let mut agent = AgentBuilder::new()
        .with_config(Config::new()
            .with_api_key(api_key)
            .with_model("gpt-4o")
            .with_max_tokens(1024))
        .with_system_prompt("You are a helpful assistant.")
        .build()?;

    // Run agent
    let response = agent.run("Tell me about Rust programming.").await?;
    println!("Response: {}", response);

    Ok(())
}
```

## Adding Tools

Agio provides two ways to add tools to your agent:

### Method 1: Implementing the RegisteredTool trait

```rust
use agio::{Agent, AgentBuilder, Config, Error};
use agio::tools::{RegisteredTool, ToolDefinition, ToolRegistry};
use async_trait::async_trait;
use serde_json::{json, Value};

struct WeatherTool;

#[async_trait]
impl RegisteredTool for WeatherTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: json!({
                "type": "object",
                "required": ["location"],
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    async fn execute(&self, arguments: Value) -> Result<String, Error> {
        let location = arguments
            .get("location")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Tool("Missing location parameter".to_string()))?;
        
        // Here you would implement the actual weather lookup logic
        // This is just a placeholder
        Ok(format!("The weather in {} is sunny and 72°F", location))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("Missing API key");
    
    // Create a tool registry and register the weather tool
    let mut tools = ToolRegistry::new();
    tools.register(WeatherTool);

    // Create agent with tools
    let mut agent = AgentBuilder::new()
        .with_config(Config::new()
            .with_api_key(api_key)
            .with_model("gpt-4o"))
        .with_system_prompt("You are a helpful assistant that can check the weather.")
        .with_tools(tools)
        .build()?;

    // Run agent
    let response = agent.run("What's the weather like in Seattle?").await?;
    println!("Response: {}", response);

    Ok(())
}
```

### Method 2: Using the function-based approach

```rust
use agio::{Agent, AgentBuilder, Config, Error, tool_fn};
use agio::tools::ToolRegistry;
use serde::{Deserialize, Serialize};

// Define the arguments schema using schemars
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct ReverseArgs {
    /// The string to reverse
    text: String,
}

// Define the function to be used as a tool
async fn reverse_string(args: ReverseArgs) -> Result<String, Error> {
    // Simply reverse the string
    let reversed: String = args.text.chars().rev().collect();
    Ok(reversed)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("Missing API key");
    
    // Create a tool registry and register the tool using the tool_fn macro
    let mut tools = ToolRegistry::new();
    
    // Option 1: Using the tool_fn macro
    let reverse_tool = tool_fn!("reverse_string", "Reverses a given string of text", reverse_string);
    tools.register(reverse_tool);
    
    // Option 2: Using the register_fn method directly
    // tools.register_fn(
    //    "reverse_string",
    //    "Reverses a given string of text",
    //    reverse_string
    // );

    // Create agent with tools
    let mut agent = AgentBuilder::new()
        .with_config(Config::new()
            .with_api_key(api_key)
            .with_model("gpt-4o"))
        .with_system_prompt("You are a helpful assistant that can reverse text.")
        .with_tools(tools)
        .build()?;

    // Run agent
    let response = agent.run("Can you reverse this text: Hello World").await?;
    println!("Response: {}", response);

    Ok(())
}
```

## Configuration Options

The library offers various configuration options:

```rust
// Example of configuring the OpenAI client
fn configure_client() -> Config {
    let config = Config::new()
        .with_api_key("your-api-key")
        .with_model("gpt-4o")         // Choose your model
        .with_temperature(0.7)        // Control randomness (0.0-2.0)
        .with_max_tokens(1024)        // Limit token generation
        .with_timeout(std::time::Duration::from_secs(30))  // Request timeout
        .with_base_url("https://api.openai.com/v1")  // API endpoint 
        .with_organization("your-org-id")  // Optional organization
        .with_stream(false)           // Enable/disable streaming
        .with_json_mode(false);       // Enable/disable JSON mode

    config
}
```

## WebSocket Support for OpenAI "Realtime" Beta

Agio includes support for OpenAI's "Realtime" Beta API via WebSockets:

```rust
use agio::{Agent, AgentBuilder, Config, Error};
use agio::websocket_client::{RealtimeEvent, ServerEvent};
use agio::tools::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("Missing API key");
    
    // Setup your tools registry
    let mut registry = ToolRegistry::new();
    // ... register your tools here
    
    // Create an agent with WebSocket support
    let mut agent = AgentBuilder::new()
        .with_config(Config::new()
            .with_api_key(api_key)
            .with_model("gpt-4o"))
        .with_tools(registry)
        .with_system_prompt("You are a helpful assistant.")
        .with_websocket()?  // Initialize WebSocket client
        .build()?;

    // Connect to the Realtime API
    agent.connect_realtime("gpt-4-realtime-preview").await?;
    
    // Send a custom event
    agent.send_realtime_event(&RealtimeEvent {
        r#type: "user_message".to_string(),
        // ... additional fields as needed
    }).await?;
    
    // Process incoming events
    agent.process_realtime_events(|event: ServerEvent| {
        println!("Received event: {:?}", event);
        Ok(())
    }).await?;
    
    // Close the connection when done
    agent.close_realtime().await?;

    Ok(())
}
```

## Error Handling

The library uses the `thiserror` crate to provide detailed error types:

```rust
// Example of handling different error types
fn handle_response() -> Result<(), Box<dyn std::error::Error>> {
    let result = agent.run("Hello").await;

    match result {
        Ok(response) => println!("Response: {}", response),
        Err(err) => match err {
            Error::Request(msg) => eprintln!("Request error: {}", msg),
            Error::Tool(msg) => eprintln!("Tool execution error: {}", msg),
            Error::Config(msg) => eprintln!("Configuration error: {}", msg),
            Error::Parse(msg) => eprintln!("Parse error: {}", msg),
            Error::Agent(msg) => eprintln!("Agent error: {}", msg),
            _ => eprintln!("Other error: {}", err),
        },
    }

    Ok(())
}
```

## Multi-turn Conversations

The agent maintains conversation state automatically:

```rust
// Example of a multi-turn conversation
async fn multi_turn_example() -> Result<(), Error> {
    // Agent setup code would be here...

    // First turn
    let response1 = agent.run("Hello, who are you?").await?;
    println!("Response 1: {}", response1);

    // Second turn (context is maintained)
    let response2 = agent.run("What did I just ask you?").await?;
    println!("Response 2: {}", response2);

    Ok(())
}
```

## Token Management

The library includes utilities for token counting:

```rust
// Example of using token management utilities
fn token_utilities_example() -> Result<(), Error> {
    use agio::utils::count_tokens;
    use agio::utils::truncate_text_to_tokens;

    // Count tokens in text
    let text = "This is a sample text";
    let token_count = count_tokens(text, "gpt-4o")?;
    println!("Token count: {}", token_count);

    // Truncate text to fit within token limits
    let long_text = "A very long text that might exceed token limits...";
    let truncated = truncate_text_to_tokens(long_text, 50, "gpt-4o")?;
    println!("Truncated text: {}", truncated);
    
    Ok(())
}
```

## Implementation Notes

- There's built-in retry logic for transient API failures.
- You can customize the agent's maximum conversation turns to prevent infinite loops.
- WebSocket support for the OpenAI "Realtime" Beta API provides a foundation for real-time interaction.

## Author

This library was created by Nikolas Yanek-Chrones (research@icarai.io).
