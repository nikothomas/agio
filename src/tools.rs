//! Tools framework for implementing agent capabilities.
//!
//! This module provides the infrastructure for defining, registering, and executing
//! tools that can be called by language models during conversation. Tools provide
//! the agent with capabilities to interact with external systems or perform specific tasks.

use crate::error::OpenAIAgentError;
use crate::models::{ToolDefinition, ToolSpec};
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

/// Trait for tools that can be registered with the agent.
///
/// Implementors of this trait define both the metadata needed for the OpenAI API
/// to understand the tool's capabilities and the actual execution logic.
#[async_trait]
pub trait RegisteredTool: Send + Sync {
    /// Returns the tool's metadata definition.
    ///
    /// This is used by the OpenAI API to understand what the tool does
    /// and what parameters it accepts.
    fn definition(&self) -> ToolDefinition;

    /// Executes the tool with the provided arguments.
    ///
    /// # Arguments
    ///
    /// * `arguments` - JSON value containing the arguments from the model
    ///
    /// # Returns
    ///
    /// A Result containing either the tool's output as a string or an error
    async fn execute(&self, arguments: Value) -> Result<String, OpenAIAgentError>;
}

/// A wrapper that turns a function into a tool.
///
/// This struct adapts a function to the `RegisteredTool` trait, automatically
/// generating the appropriate definition from the function's parameter type.
pub struct FunctionTool<F, Args, Fut, R>
where
    F: Fn(Args) -> Fut + Send + Sync,
    Args: DeserializeOwned + Serialize + Debug + Send + Sync,
    Fut: Future<Output = Result<R, OpenAIAgentError>> + Send,
    R: ToString + Send + Sync, // Added Sync bound
{
    name: String,
    description: String,
    function: F,
    _args: PhantomData<Args>,
    _fut: PhantomData<Fut>,
    _result: PhantomData<R>,
}

impl<F, Args, Fut, R> FunctionTool<F, Args, Fut, R>
where
    F: Fn(Args) -> Fut + Send + Sync,
    Args: DeserializeOwned + Serialize + Debug + Send + Sync + 'static,
    Fut: Future<Output = Result<R, OpenAIAgentError>> + Send,
    R: ToString + Send + Sync, // Added Sync bound
{
    /// Creates a new FunctionTool from a function and metadata.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool
    /// * `description` - A description of what the tool does
    /// * `function` - The function to execute
    ///
    /// # Returns
    ///
    /// A new FunctionTool instance
    pub fn new(name: impl Into<String>, description: impl Into<String>, function: F) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            function,
            _args: PhantomData,
            _fut: PhantomData,
            _result: PhantomData,
        }
    }
}
// Inside tools.rs, update the definition method in the RegisteredTool impl for FunctionTool:

#[async_trait]
impl<F, Args, Fut, R> RegisteredTool for FunctionTool<F, Args, Fut, R>
where
    F: Fn(Args) -> Fut + Send + Sync + 'static + std::clone::Clone,
    Args: DeserializeOwned + Serialize + Debug + Send + Sync + 'static + schemars::JsonSchema,
    Fut: Future<Output = Result<R, OpenAIAgentError>> + Send + 'static + std::marker::Sync,
    R: ToString + Send + Sync + 'static,
{
    fn definition(&self) -> ToolDefinition {
        // Generate schema from Args type using schemars
        let schema = schemars::schema_for!(Args);

        // Convert to JSON value and modify it to ensure additionalProperties is false
        let mut schema_value = serde_json::to_value(schema).unwrap_or_else(|_| {
            serde_json::json!({
                "type": "object",
                "properties": {},
            })
        });

        // Ensure schema is an object
        if let Some(schema_obj) = schema_value.as_object_mut() {
            // Explicitly add additionalProperties: false at the top level
            schema_obj.insert(
                "additionalProperties".to_string(),
                serde_json::Value::Bool(false)
            );
        }

        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: schema_value,
            strict: Some(true),
        }
    }

    // Rest of the implementation remains the same...
    async fn execute(&self, arguments: Value) -> Result<String, OpenAIAgentError> {
        // Clone the values we need to move into the async block to prevent lifetime issues
        let function = self.function.clone();

        // Parse the arguments into the expected type
        let args: Args = serde_json::from_value(arguments.clone())
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to parse arguments: {}", e)))?;

        // Call the function with owned values
        let result = function(args).await?;

        // Convert the result to a string
        Ok(result.to_string())
    }
}

/// Registry for managing the tools available to an agent.
///
/// This struct stores all the tools that can be used by the agent,
/// allowing lookup by name and providing metadata for API requests.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    /// Map of tool names to their implementations
    tools: HashMap<String, Arc<dyn RegisteredTool>>,
}

impl ToolRegistry {
    /// Creates an empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a new tool with the registry.
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool implementation to register
    pub fn register<T>(&mut self, tool: T)
    where
        T: RegisteredTool + 'static,
    {
        let definition = tool.definition();
        self.tools.insert(definition.name.clone(), Arc::new(tool));
    }

    /// Registers a function as a tool.
    ///
    /// This is a convenience method that creates a FunctionTool and registers it.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool
    /// * `description` - A description of what the tool does
    /// * `function` - The function to execute
    ///
    /// # Returns
    ///
    /// A reference to self for method chaining
    pub fn register_fn<F, Args, Fut, R>(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        function: F,
    ) -> &mut Self
    where
        F: Fn(Args) -> Fut + Send + Sync + Clone + 'static,
        Args: DeserializeOwned + Serialize + Debug + Send + Sync + 'static + schemars::JsonSchema,
        Fut: Future<Output = Result<R, OpenAIAgentError>> + Send + 'static + std::marker::Sync,
        R: ToString + Send + Sync + 'static,
    {
        let tool = FunctionTool::new(name, description, function);
        self.register(tool);
        self
    }

    /// Looks up a tool by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool to look up
    ///
    /// # Returns
    ///
    /// An Option containing a reference to the tool if found
    pub fn get(&self, name: &str) -> Option<Arc<dyn RegisteredTool>> {
        self.tools.get(name).cloned()
    }

    /// Checks if the registry is empty.
    ///
    /// # Returns
    ///
    /// `true` if there are no registered tools, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Returns the definitions of all tools in the proper format for API requests.
    ///
    /// # Returns
    ///
    /// A vector of ToolSpec objects containing the tool definitions
    pub fn definitions(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|t| {
                let def = t.definition();
                ToolSpec {
                    r#type: "function".to_string(),
                    function: def,
                }
            })
            .collect()
    }
}