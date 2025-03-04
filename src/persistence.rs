//! Persistence layer for storing and retrieving agent data.
//!
//! This module provides traits and implementations for persisting agent state,
//! conversations, and other data to various storage backends.

use crate::agent::AgentState;
use crate::error::OpenAIAgentError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Unique identifier for stored entities
pub type EntityId = String;

/// Generate a new unique ID
pub fn generate_id() -> EntityId {
    Uuid::new_v4().to_string()
}

/// Metadata for stored conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMetadata {
    /// Unique identifier
    pub id: EntityId,
    /// Optional user-provided name
    pub name: Option<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Number of messages
    pub message_count: usize,
    /// Total tokens used
    pub token_count: usize,
}

/// Core persistence trait for storing and retrieving agent data
#[async_trait]
pub trait PersistenceStore: Send + Sync {
    /// Store a conversation with its messages
    async fn store_conversation(&self, id: &str, state: &AgentState) -> Result<(), OpenAIAgentError>;
    
    /// Retrieve a conversation by ID
    async fn get_conversation(&self, id: &str) -> Result<Option<AgentState>, OpenAIAgentError>;
    
    /// Delete a conversation
    async fn delete_conversation(&self, id: &str) -> Result<(), OpenAIAgentError>;
    
    /// List available conversations with metadata
    async fn list_conversations(&self, limit: usize, offset: usize) -> Result<Vec<ConversationMetadata>, OpenAIAgentError>;
}

// Re-export implementations
pub mod memory;
#[cfg(feature = "postgres")]
pub mod postgres;

// Re-export implementations for easier access
pub use memory::MemoryStore;
#[cfg(feature = "postgres")]
pub use postgres::PostgresStore; 