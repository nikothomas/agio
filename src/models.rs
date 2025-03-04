//! Data models for OpenAI API requests and responses.
//!
//! This module defines the structures used to represent messages, requests,
//! responses, and other data related to the OpenAI API, with particular
//! focus on chat completions and tool calls.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A message in a conversation with various roles (system, user, assistant, tool).
///
/// Messages form the core of chat interactions with OpenAI models. Each message
/// has a role, optional content, and may include tool calls for agent functionality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender (system, user, assistant, or tool)
    pub role: String,

    /// Content of the message, optional when using tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Name of the speaker if applicable (e.g., tool name)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Required for 'tool' role messages - must match the id of the tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Tool calls array for OpenAI's newer API format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    /// Creates a system message.
    ///
    /// # Arguments
    ///
    /// * `content` - The content of the system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Creates a user message.
    ///
    /// # Arguments
    ///
    /// * `content` - The content of the user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Creates an assistant message.
    ///
    /// # Arguments
    ///
    /// * `content` - The content of the assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Creates a tool result message.
    ///
    /// # Arguments
    ///
    /// * `content` - The result of the tool execution
    /// * `tool_name` - The name of the tool that was executed
    /// * `tool_call_id` - The ID of the tool call this result is for
    pub fn tool_result(content: impl Into<String>, tool_name: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            name: Some(tool_name.into()),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

/// Metadata for a tool that can be called by the model.
///
/// This struct defines a tool's name, description, and parameters schema
/// for the OpenAI API to understand its capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Name of the tool
    pub name: String,

    /// Description of what the tool does
    pub description: String,

    /// JSON Schema defining the tool's parameters
    pub parameters: serde_json::Value,

    /// Optional flag for strict JSON schema enforcement
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Top-level wrapper for a tool in the API request.
///
/// The OpenAI API expects tools to be wrapped with a type field and
/// nested function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolSpec {
    /// Type of the tool (currently always "function")
    #[serde(rename = "type")]
    pub r#type: String,

    /// The tool definition
    pub function: ToolDefinition,
}

/// Function data within a tool call.
///
/// Contains the name of the function to call and its arguments as a JSON string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct FunctionCall {
    /// Name of the function to call
    #[serde(default)]
    pub name: String,

    /// Arguments for the function as a JSON string
    #[serde(default)]
    pub arguments: String,
}

/// A request from the model to call a tool.
///
/// This struct represents the model's request to execute a tool,
/// including the tool name and arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    #[serde(default)]
    pub id: String,

    /// Type of the call (always "function" in current API)
    #[serde(rename = "type", default)]
    pub call_type: String,

    /// Function information in newer API format
    #[serde(default)]
    pub function: FunctionCall,

    /// Name of the tool (older API format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Arguments as a JSON string (older API format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

impl ToolCall {
    /// Helper to get the name regardless of API format.
    ///
    /// # Returns
    ///
    /// The name of the tool to call
    pub fn get_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.function.name.clone())
    }

    /// Helper to get the arguments regardless of API format.
    ///
    /// # Returns
    ///
    /// The arguments as a JSON string
    pub fn get_arguments(&self) -> String {
        self.arguments.clone().unwrap_or_else(|| self.function.arguments.clone())
    }
}

/// Request to the OpenAI Chat Completions API.
///
/// This struct contains all the parameters for a chat completion request,
/// including the model, messages, tools, and generation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatRequest {
    /// Model identifier to use for completion
    pub model: String,

    /// Conversation history as a sequence of messages
    pub messages: Vec<ChatMessage>,

    /// Tools that the model can use during the conversation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,

    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Temperature for controlling randomness
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Format for the response (e.g., {"type": "json_object"})
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<HashMap<String, String>>,

    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// A single choice/response from the model.
///
/// Represents one possible completion from the model,
/// including the generated message and finish reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatChoice {
    /// Index of this choice in the array of choices
    pub index: usize,

    /// The message generated by the model
    pub message: ChatMessage,

    /// Tool calls at the choice level (newer API format)
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,

    /// Reason why the model stopped generating
    pub finish_reason: String,
}

/// Response from the OpenAI Chat Completions API.
///
/// This struct contains the complete response from the API,
/// including all generated choices and token usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatResponse {
    /// Unique identifier for this completion
    pub id: String,

    /// Object type (always "chat.completion")
    pub object: String,

    /// Timestamp when the completion was created
    pub created: u64,

    /// Model used for the completion
    pub model: String,

    /// Array of completion choices
    pub choices: Vec<ChatChoice>,

    /// Token usage statistics
    pub usage: Option<Usage>,
}

/// Token usage statistics for a request/response.
///
/// This struct tracks the number of tokens used in the prompt,
/// completion, and in total for billing purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,

    /// Number of tokens in the completion
    pub completion_tokens: usize,

    /// Total number of tokens used
    pub total_tokens: usize,
}

// Public interfaces - only expose what users actually need directly
/// Public message interface for users who need to work with messages
#[cfg(feature = "experimental")]
#[derive(Debug, Clone)]
pub struct Message {
    inner: ChatMessage
}

#[cfg(feature = "experimental")]
impl Message {
    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self { inner: ChatMessage::system(content) }
    }

    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self { inner: ChatMessage::user(content) }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { inner: ChatMessage::assistant(content) }
    }

    /// Get the role of this message
    pub fn role(&self) -> &str {
        &self.inner.role
    }

    /// Get the content of this message, if any
    pub fn content(&self) -> Option<&str> {
        self.inner.content.as_deref()
    }
}