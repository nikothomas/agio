//! agent.rs
//!
//! Agent module for managing conversational interactions with OpenAI models
//! via both HTTP and (optionally) WebSocket "Realtime" Beta.

use crate::client::OpenAIClient;
use crate::config::OpenAIConfig;
use crate::error::OpenAIAgentError;
use crate::models::{ChatMessage, ChatRequest, ChatResponse, ToolCall};
use crate::tools::ToolRegistry;
use crate::websocket_client::{WebSocketClient, RealtimeEvent, ServerEvent};
#[cfg(feature = "memory")]
use crate::persistence::{EntityId, PersistenceStore, generate_id};
use std::sync::Arc;

/// The current state of the agent, including conversation history and token usage.
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Conversation history including all messages exchanged
    pub(crate) messages: Vec<ChatMessage>,

    /// Running count of tokens used in the conversation
    pub(crate) token_count: usize,
}

impl AgentState {
    /// Returns the total number of tokens used in the conversation so far
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Returns the number of messages in the conversation history
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Returns an iterator over references to the messages in the conversation
    pub fn messages(&self) -> impl Iterator<Item = &ChatMessage> {
        self.messages.iter()
    }
}

/// An agent that manages conversations with OpenAI models.
///
/// The agent handles the conversation flow, including sending requests to the API,
/// processing responses, and optionally executing tool calls when the model requests them.
/// It can also connect to the (hypothetical) OpenAI "Realtime" Beta API over WebSockets.
pub struct Agent {
    /// OpenAI client for making normal HTTP API requests
    client: OpenAIClient,

    /// Tools that the agent can use when prompted by the model
    tools: Arc<ToolRegistry>,

    /// Current state of the agent including message history
    state: AgentState,

    /// Maximum number of turns before terminating to prevent infinite loops
    max_turns: usize,

    /// Optional WebSocket client for the OpenAI "Realtime" Beta API
    websocket_client: Option<WebSocketClient>,
    
    /// Unique identifier for this agent instance
    #[cfg(feature = "memory")]
    id: EntityId,
    
    /// Optional persistence store
    #[cfg(feature = "memory")]
    persistence: Option<Arc<dyn PersistenceStore>>,
}

impl Agent {
    /// Creates a new agent from an `AgentBuilder` configuration.
    ///
    /// This method is intended for internal use by the `AgentBuilder`.
    #[doc(hidden)]
    pub(crate) fn from_builder(builder: AgentBuilder) -> Result<Self, OpenAIAgentError> {
        let client = OpenAIClient::new(builder.config.clone().unwrap_or_default())?;

        let state = AgentState {
            messages: builder.messages,
            token_count: 0,
        };

        #[cfg(not(feature = "memory"))]
        let agent = Self {
            client,
            tools: builder.tools,
            state,
            max_turns: builder.max_turns,
            websocket_client: builder.websocket_client,
        };
        
        #[cfg(feature = "memory")]
        let agent = Self {
            client,
            tools: builder.tools,
            state,
            max_turns: builder.max_turns,
            websocket_client: builder.websocket_client,
            id: builder.id,
            persistence: builder.persistence,
        };

        Ok(agent)
    }

    /// Primary entry point for user → agent → OpenAI conversation using HTTP-based chat completions.
    ///
    /// If you plan to use the WebSocket "Realtime" approach, you might either
    /// not use this method or adapt it to handle real-time streaming directly.
    pub async fn run(&mut self, input: impl Into<String>) -> Result<String, OpenAIAgentError> {
        let result = self.run_internal(input).await?;
        
        // Optionally save state after each interaction
        #[cfg(feature = "memory")]
        if self.persistence.is_some() {
            self.save().await?;
        }
        
        Ok(result)
    }
    
    /// Internal implementation of run that doesn't save state
    async fn run_internal(&mut self, input: impl Into<String>) -> Result<String, OpenAIAgentError> {
        self.state.messages.push(ChatMessage::user(input.into()));

        let mut turns = 0;
        let mut final_response = String::new();

        while turns < self.max_turns {
            turns += 1;

            let request = self.prepare_request()?;
            let response = self.client.chat_completion(request).await?;

            if let Some(usage) = response.usage.as_ref() {
                self.state.token_count += usage.total_tokens;
            }

            if let Some(choice) = response.choices.first() {
                self.state.messages.push(choice.message.clone());

                if let Some(tool_calls) = &choice.message.tool_calls {
                    if !tool_calls.is_empty() {
                        // Process each tool call
                        for tool_call in tool_calls {
                            let result_msg = self.execute_tool_call(tool_call).await?;
                            self.state.messages.push(result_msg);
                        }

                        // Once we've processed tool calls, go back to top of the loop
                        continue;
                    }
                }

                // If there's direct content, return it
                if let Some(content) = &choice.message.content {
                    if !content.trim().is_empty() {
                        final_response = content.clone();
                        return Ok(final_response);
                    }
                }

                // If the finish reason was "tool_calls", we might want to continue
                if choice.finish_reason == "tool_calls" {
                    continue;
                }

                return Err(OpenAIAgentError::Parse(
                    format!("Assistant returned empty message with finish_reason: {}", choice.finish_reason),
                ));
            }
            return Err(OpenAIAgentError::Parse(
                "No response choices received".to_string(),
            ));
        }

        Err(OpenAIAgentError::Agent(format!(
            "Agent exceeded maximum turns ({})",
            self.max_turns
        )))
    }

    /// Internal helper that executes a given tool call (function call).
    async fn execute_tool_call(&self, tc: &ToolCall) -> Result<ChatMessage, OpenAIAgentError> {
        let tool_name = &tc.function.name;
        let arguments = &tc.function.arguments;
        let tool_call_id = &tc.id;

        let tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| OpenAIAgentError::Tool(format!("Tool not found: {}", tool_name)))?;

        // Parse the JSON arguments
        let parsed_args = serde_json::from_str(arguments)
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to parse tool arguments: {}", e)))?;

        // Execute the tool
        let result = tool.execute(parsed_args).await?;

        // Create a message that records the tool's result
        let response = ChatMessage {
            role: "tool".to_string(),
            content: Some(result),
            name: Some(tool_name.clone()),
            tool_call_id: Some(tool_call_id.clone()),
            tool_calls: None,
        };

        Ok(response)
    }

    /// Prepares a request to the OpenAI API with the current state and tools.
    fn prepare_request(&self) -> Result<ChatRequest, OpenAIAgentError> {
        let config = self.client.config();

        let mut request = ChatRequest {
            model: config.model().to_string(),
            messages: self.state.messages.clone(),
            tools: None,
            max_tokens: Some(config.max_tokens()),
            temperature: Some(config.temperature()),
            response_format: None,
            stream: Some(config.stream()),
        };

        if !self.tools.is_empty() {
            request.tools = Some(self.tools.definitions());
        }

        Ok(request)
    }

    /// Returns a reference to the current agent state.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Adds a user message to the conversation history.
    ///
    /// # Arguments
    ///
    /// * `content` - The content of the user message
    pub fn push_user_message(&mut self, content: impl Into<String>) {
        self.state.messages.push(ChatMessage::user(content.into()));
    }

    /// Adds an assistant message to the conversation history.
    ///
    /// # Arguments
    ///
    /// * `content` - The content of the assistant message
    pub fn push_assistant_message(&mut self, content: impl Into<String>) {
        self.state.messages.push(ChatMessage::assistant(content.into()));
    }
    
    /// Get the agent's unique identifier
    #[cfg(feature = "memory")]
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Save the current agent state to the persistence store
    #[cfg(feature = "memory")]
    pub async fn save(&self) -> Result<(), OpenAIAgentError> {
        if let Some(store) = &self.persistence {
            store.store_conversation(&self.id, &self.state).await?;
        }
        Ok(())
    }
    
    /// Load agent state from the persistence store
    #[cfg(feature = "memory")]
    pub async fn load(&mut self) -> Result<bool, OpenAIAgentError> {
        if let Some(store) = &self.persistence {
            if let Some(state) = store.get_conversation(&self.id).await? {
                self.state = state;
                return Ok(true);
            }
        }
        Ok(false)
    }
    
    /// Delete agent data from the persistence store
    #[cfg(feature = "memory")]
    pub async fn delete(&self) -> Result<(), OpenAIAgentError> {
        if let Some(store) = &self.persistence {
            store.delete_conversation(&self.id).await?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Below are the NEW methods enabling WebSocket usage ("Realtime" Beta)
    // ------------------------------------------------------------------------

    /// Connect to the "Realtime" WebSocket API, specifying which model to use.
    pub async fn connect_realtime(&mut self, model_name: &str) -> Result<(), OpenAIAgentError> {
        let ws_client = self
            .websocket_client
            .as_mut()
            .ok_or_else(|| OpenAIAgentError::Agent(
                "No WebSocket client configured (call `with_websocket()` first).".to_string()
            ))?;

        ws_client.connect(model_name).await
    }

    /// Send a custom event/message to the Realtime API over the WebSocket.
    pub async fn send_realtime_event(
        &mut self,
        event: &RealtimeEvent
    ) -> Result<(), OpenAIAgentError> {
        let ws_client = self
            .websocket_client
            .as_mut()
            .ok_or_else(|| OpenAIAgentError::Agent(
                "No WebSocket client configured (call `with_websocket()` first).".to_string()
            ))?;

        ws_client.send_event(event).await
    }

    /// Start reading events from the Realtime API in a loop, calling the provided handler for each.
    ///
    /// The handler can do whatever you like with inbound `ServerEvent`s,
    /// e.g., store them in your conversation, feed them into the agent, etc.
    pub async fn process_realtime_events<F>(
        &mut self,
        on_event: F
    ) -> Result<(), OpenAIAgentError>
    where
        F: FnMut(ServerEvent) -> Result<(), OpenAIAgentError>
    {
        let ws_client = self
            .websocket_client
            .as_mut()
            .ok_or_else(|| OpenAIAgentError::Agent(
                "No WebSocket client configured (call `with_websocket()` first).".to_string()
            ))?;

        ws_client.process_incoming(on_event).await
    }

    /// Close the WebSocket connection gracefully, if open.
    pub async fn close_realtime(&mut self) -> Result<(), OpenAIAgentError> {
        if let Some(ws_client) = &mut self.websocket_client {
            ws_client.close().await?;
        }
        Ok(())
    }
}

/// Builder for creating Agent instances with a fluent interface.
pub struct AgentBuilder {
    /// OpenAI configuration
    pub(crate) config: Option<OpenAIConfig>,

    /// Tools available to the agent
    pub(crate) tools: Arc<ToolRegistry>,

    /// Initial messages to seed the conversation
    pub(crate) messages: Vec<ChatMessage>,

    /// Maximum number of turns to prevent infinite loops
    pub(crate) max_turns: usize,

    /// Optional WebSocket client for the Realtime Beta
    pub(crate) websocket_client: Option<WebSocketClient>,
    
    /// Unique identifier for the agent
    #[cfg(feature = "memory")]
    pub(crate) id: EntityId,
    
    /// Optional persistence store
    #[cfg(feature = "memory")]
    pub(crate) persistence: Option<Arc<dyn PersistenceStore>>,
}

impl AgentBuilder {
    /// Creates a new AgentBuilder with default settings.
    pub fn new() -> Self {
        #[cfg(not(feature = "memory"))]
        let builder = Self {
            config: None,
            tools: Arc::new(ToolRegistry::new()),
            messages: Vec::new(),
            max_turns: 10,
            websocket_client: None,
        };
        
        #[cfg(feature = "memory")]
        let builder = Self {
            config: None,
            tools: Arc::new(ToolRegistry::new()),
            messages: Vec::new(),
            max_turns: 10,
            websocket_client: None,
            id: generate_id(),
            persistence: None,
        };
        
        builder
    }

    /// Sets the OpenAI configuration.
    pub fn with_config(mut self, config: OpenAIConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Sets the tool registry.
    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Arc::new(tools);
        self
    }

    /// Adds a system prompt message to the initial conversation.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::system(prompt.into()));
        self
    }

    /// Adds a message to the initial conversation.
    pub fn with_message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Sets the maximum number of conversation turns.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }
    
    /// Set a specific ID for the agent
    #[cfg(feature = "memory")]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
    
    /// Add persistence capabilities to the agent
    #[cfg(feature = "memory")]
    pub fn with_persistence(mut self, store: Arc<dyn PersistenceStore>) -> Self {
        self.persistence = Some(store);
        self
    }

    /// Instantiates a WebSocketClient for Realtime usage, storing it in this builder.
    /// This does NOT immediately connect; call `agent.connect_realtime(...)` after build.
    pub fn with_websocket(mut self) -> Result<Self, OpenAIAgentError> {
        let cfg = self.config.clone().unwrap_or_default();
        let ws_client = WebSocketClient::new(cfg)?;
        self.websocket_client = Some(ws_client);
        Ok(self)
    }

    /// Builds the Agent from the current configuration.
    pub fn build(self) -> Result<Agent, OpenAIAgentError> {
        Agent::from_builder(self)
    }
    
    /// Build the agent, optionally loading state from persistence
    #[cfg(feature = "memory")]
    pub async fn build_async(self) -> Result<Agent, OpenAIAgentError> {
        let mut agent = Agent::from_builder(self)?;
        
        // If persistence is configured, try to load existing state
        if agent.persistence.is_some() {
            let _ = agent.load().await?; // Ignore if no state exists yet
        }
        
        Ok(agent)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
