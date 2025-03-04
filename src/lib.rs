//! A Rust crate for configuring and using OpenAI in an agentic system.
//!
//! This library provides a high-level interface to interact with OpenAI's API,
//! with particular focus on agent-based interactions that leverage tool calls
//! (formerly known as function calls).
//!
//! # Basic usage
//!
//! ```rust
//! use agio::{Agent, AgentBuilder, Config};
//!
//! # async fn example() -> Result<(), agio::Error> {
//! // Create a configuration
//! let config = Config::new()
//!     .with_api_key("your-api-key")
//!     .with_model("gpt-4o");
//!
//! // Create an agent
//! let mut agent = AgentBuilder::new()
//!     .with_config(config)
//!     .with_system_prompt("You are a helpful assistant.")
//!     .build()?;
//!
//! // Run the agent
//! let response = agent.run("Tell me about Rust programming.").await?;
//! println!("Response: {}", response);
//! # Ok(())
//! # }
//! ```

// Internal modules
mod agent;
mod config;
mod client;
mod error;
mod utils;
mod tools;
pub mod websocket_client;

// Persistence and server modules
pub mod persistence;
pub mod server;
pub mod models;

// Public exports for the prelude
pub mod prelude {
    //! Commonly used types and traits
    //!
    //! This module re-exports the most commonly used types and traits from the crate
    //! to make them more easily accessible.

    pub use crate::agent::{Agent, AgentBuilder};
    pub use crate::config::OpenAIConfig as Config;
    pub use crate::error::OpenAIAgentError as Error;
    pub use crate::tools::{ToolRegistry, RegisteredTool};
    
    // Re-export persistence types
    pub use crate::persistence::{PersistenceStore, EntityId, ConversationMetadata, MemoryStore};
    pub use crate::persistence::postgres::PostgresStore;
    
    // Re-export server types
    pub use crate::server::AgentManager;
}

// Direct exports for the main API surface
pub use prelude::*;

// Re-export from models for public use
pub use crate::models::ToolDefinition;

// Re-export FunctionTool
pub use crate::tools::FunctionTool;

// Selective re-exports of internal types that are needed in public APIs
// but should not be directly constructed by users
pub use agent::AgentState;

// Explicitly re-export persistence and server modules
pub use persistence::{PersistenceStore, EntityId, ConversationMetadata, MemoryStore};
pub use persistence::postgres::PostgresStore;
pub use server::AgentManager;

// Define the tool_fn macro directly in lib.rs to avoid module path issues
/// Creates a tool from a function.
///
/// This macro simplifies creating tools from functions by handling the type inference.
///
/// # Arguments
///
/// * `name` - The name of the tool
/// * `description` - A description of what the tool does
/// * `function` - The function to execute
///
/// # Example
///
/// ```
/// use agio::tool_fn;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
/// struct ReverseArgs {
///     text: String,
/// }
///
/// async fn reverse_string(args: ReverseArgs) -> Result<String, agio::Error> {
///     Ok(args.text.chars().rev().collect())
/// }
///
/// let tool = tool_fn!("reverse_string", "Reverses a string", reverse_string);
/// ```
#[macro_export]
macro_rules! tool_fn {
    ($name:expr, $description:expr, $function:expr) => {
        $crate::FunctionTool::new($name, $description, $function)
    };
}