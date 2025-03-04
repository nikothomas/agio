// tests/integration_test.rs

use std::env;
use tokio::runtime::Runtime;

// Import only from the public API
use agio::{
    AgentBuilder, Config, Error,
    RegisteredTool, ToolRegistry, ToolDefinition,
    tool_fn // Now properly exported in lib.rs
};

use async_trait::async_trait;
use rustls::crypto::CryptoProvider;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

// Method 1: Traditional implementation with the RegisteredTool trait
pub struct ReverseToolTraditional;

#[async_trait]
impl RegisteredTool for ReverseToolTraditional {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "reverse_string_traditional".to_string(),
            description: "Reverses a given string of text (traditional implementation).".to_string(),
            strict: Some(true),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The string to reverse."
                    }
                },
                "required": ["text"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: Value) -> Result<String, Error> {
        let text = arguments
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Tool("Missing 'text' argument".to_string()))?;

        // For the test, we just return the actual reversed string
        let reversed: String = text.chars().rev().collect();

        Ok(reversed)
    }
}

// Method 2: Function-based approach
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

// ──────────────────────────────────────────────────────────────────────────────
// TESTS
// ──────────────────────────────────────────────────────────────────────────────

// This test checks that both “traditional” and “function-based” approaches
// can call tools and reverse a random string (UUID).
#[test]
fn test_random_string_reverse_both_approaches() -> Result<(), Box<dyn std::error::Error>> {
    // 1) Get your OPENAI_API_KEY from an environment variable
    let api_key = env::var("OPENAI_API_KEY")
        .expect("Set the OPENAI_API_KEY env var before running this test.");

    // 2) Create a random string (UUID) that the model can't predict
    let random_uuid = Uuid::new_v4().to_string();
    let reversed_uuid: String = random_uuid.chars().rev().collect();

    // 3) Build a Tokio runtime
    let rt = Runtime::new()?;

    // 4) Test traditional approach
    let traditional_test = rt.block_on(async {
        // Directly use the async function without BoxFuture wrapper
        let registry = setup_traditional_tools().await;
        test_approach(
            "traditional",
            &api_key,
            &random_uuid,
            &reversed_uuid,
            registry
        ).await
    })?;

    // 5) Test function-based approach
    let function_test = rt.block_on(async {
        // Directly use the async function without BoxFuture wrapper
        let registry = setup_function_tools().await;
        test_approach(
            "function-based",
            &api_key,
            &random_uuid,
            &reversed_uuid,
            registry
        ).await
    })?;

    Ok(())
}

// Optional: test usage of your new WebSocket-based “Realtime” flow
// This test will ONLY pass if your key and org have access to the Beta and
// you have a valid model name that supports Realtime. Otherwise, you may
// want to remove or ignore this test.
#[test]
fn test_websocket_realtime_reverse() -> Result<(), Box<dyn std::error::Error>> {
    rustls::crypto::ring::default_provider().install_default().unwrap();
    let api_key = env::var("OPENAI_API_KEY")
        .expect("Set the OPENAI_API_KEY env var before running this test.");

    // 2) Create random text
    let random_uuid = Uuid::new_v4().to_string();
    let reversed_uuid: String = random_uuid.chars().rev().collect();

    // 3) Build a Tokio runtime
    let rt = Runtime::new()?;

    // 4) Run asynchronously
    rt.block_on(async {
        // a) Create a registry with your reverse tool
        let registry = setup_traditional_tools().await;
        // or use setup_function_tools() if you prefer

        // b) Build config
        let config = Config::new()
            .with_api_key(api_key)
            .with_model("gpt-3.5-turbo") // Fallback for normal calls if needed
            .with_max_tokens(300)
            .with_temperature(0.0);

        // c) Create an agent that also initializes a WebSocketClient
        let mut agent = AgentBuilder::new()
            .with_config(config)
            .with_tools(registry)
            .with_system_prompt(r#"You are a strict Reverser Assistant (Realtime test).
                If the user requests text reversal, you MUST call the 'reverse_string_traditional' tool.
                "#.to_string())
            .with_websocket()?  // <-- new line that creates a WebSocketClient
            .build()?;

        // d) Connect to the Realtime API (adjust model to whatever Beta name you have)
        agent.connect_realtime("gpt-4-realtime-preview").await?;
        println!("Connected to Realtime API over WebSocket.");

        // e) Possibly send a “hello” event or start reading events
        // This is optional, just to show usage:
        use agio::websocket_client::RealtimeEvent;
        agent.send_realtime_event(&RealtimeEvent {
            r#type: "test_hello".to_string(),
            // Any custom fields you may have in your MyRealtimeEvent struct...
        }).await?;

        // g) Now we can still do a normal “run” if we want to test the tool calls
        //   (Though it’ll use HTTP, not Realtime, unless you redesign your .run method.)
        let user_msg = format!("Please reverse this text: {}", random_uuid);
        println!("(Realtime) Attempting normal .run with text: {}", random_uuid);

        let response = agent.run(&user_msg).await?; // This hits the REST ChatCompletion

        println!("Got response: {}", response);
        if response.contains(&reversed_uuid) {
            println!("Realtime approach test passed!");
        } else {
            return Err(Error::Agent(format!(
                "Realtime approach test failed: The final response '{response}' did not contain '{reversed_uuid}'"
            )));
        }

        // h) Clean up
        agent.close_realtime().await?;
        println!("WebSocket closed gracefully.");
        Ok(())
    })?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// HELPER SETUP
// ──────────────────────────────────────────────────────────────────────────────

// Helper function to set up traditional tools
async fn setup_traditional_tools() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReverseToolTraditional);
    registry
}

// Helper function to set up function-based tools
async fn setup_function_tools() -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Method 1: Use the register_fn method
    registry.register_fn(
        "reverse_string",
        "Reverses a given string of text (function-based implementation).",
        reverse_string,
    );

    // Alternative - Method 2: Use the tool_fn! macro
    // registry.register(tool_fn!(
    //     "reverse_string",
    //     "Reverses a given string of text (function-based implementation).",
    //     reverse_string
    // ));

    registry
}

// Simplified test function that accepts a ready registry instead of setup function
async fn test_approach(
    approach_name: &str,
    api_key: &str,
    random_uuid: &str,
    reversed_uuid: &str,
    registry: ToolRegistry
) -> Result<(), Error> {
    println!("Testing {} approach", approach_name);

    // a) Config with a Tools-calling model
    let config = Config::new()
        .with_api_key(api_key.to_string())
        .with_model("gpt-3.5-turbo")  // Try a simpler model first for testing
        .with_max_tokens(300)         // Increase token limit
        .with_temperature(0.0);       // Deterministic output for testing

    // Choose the right tool name depending on approach
    let tool_name = if approach_name == "traditional" {
        "reverse_string_traditional"
    } else {
        "reverse_string"
    };

    // c) Build the agent with a strong system prompt
    let mut agent = AgentBuilder::new()
        .with_config(config)
        .with_tools(registry)
        .with_system_prompt(
            format!(r#"You are a strict Reverser Assistant.
            If the user requests text reversal, you MUST call the '{}' tool.
            NEVER attempt to reverse text yourself.
            After receiving the reversed text from the tool, include it in your response.
            "#, tool_name)
        )
        .build()?;

    // d) Actually use the random UUID so the test is legitimate:
    let user_msg = format!("Please reverse this text: {}", random_uuid);

    println!("Sending request to reverse: {}", random_uuid);
    println!("Expected response should contain: {}", reversed_uuid);

    let response = agent.run(&user_msg).await?;

    println!("Got response: {}", response);
    // e) Check if the final output has the reversed random UUID
    if response.contains(reversed_uuid) {
        println!("{} approach test passed!", approach_name);
        Ok(())
    } else {
        Err(Error::Agent(format!(
            "{} approach test failed: The final response did not contain '{}'",
            approach_name, reversed_uuid
        )))
    }
}
