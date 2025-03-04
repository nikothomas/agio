//! Error types for the OpenAI agent library.
//!
//! This module defines a comprehensive error type that encapsulates
//! all possible error conditions that can occur during agent operation.

use thiserror::Error;

/// Comprehensive error type for OpenAI agent operations.
///
/// This enum represents all the different types of errors that can occur
/// when using the OpenAI agent library, from network issues to parsing problems.
#[derive(Debug, Error)]
pub enum OpenAIAgentError {
    /// Error occurring during HTTP request processing
    #[error("Request error: {0}")]
    Request(String),

    /// Error from the reqwest HTTP client
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    /// Error in configuration parameters
    #[error("Configuration error: {0}")]
    Config(String),

    /// JSON serialization or deserialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Error related to tool execution
    #[error("Tool error: {0}")]
    Tool(String),

    /// Error parsing data or responses
    #[error("Parse error: {0}")]
    Parse(String),

    /// Error in agent operation logic
    #[error("Agent error: {0}")]
    Agent(String),

    /// IO error from standard library
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}