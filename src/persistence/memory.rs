//! In-memory implementation of the persistence store.
//!
//! This module provides a simple in-memory implementation of the PersistenceStore
//! trait, useful for testing and development.

use super::{ConversationMetadata, EntityId, PersistenceStore};
use crate::agent::AgentState;
use crate::error::OpenAIAgentError;
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// In-memory implementation of PersistenceStore for testing and development
pub struct MemoryStore {
    conversations: Arc<RwLock<HashMap<EntityId, (AgentState, ConversationMetadata)>>>,
}

impl MemoryStore {
    /// Create a new empty memory store
    pub fn new() -> Self {
        Self {
            conversations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PersistenceStore for MemoryStore {
    async fn store_conversation(&self, id: &str, state: &AgentState) -> Result<(), OpenAIAgentError> {
        let mut conversations = self.conversations.write().map_err(|e| {
            OpenAIAgentError::Agent(format!("Failed to acquire write lock: {}", e))
        })?;
        
        let now = Utc::now();
        
        // Create or update metadata
        let metadata = if let Some((_, mut meta)) = conversations.get(id).cloned() {
            meta.updated_at = now;
            meta.message_count = state.message_count();
            meta.token_count = state.token_count();
            meta
        } else {
            ConversationMetadata {
                id: id.to_string(),
                name: None,
                created_at: now,
                updated_at: now,
                message_count: state.message_count(),
                token_count: state.token_count(),
            }
        };
        
        conversations.insert(id.to_string(), (state.clone(), metadata));
        Ok(())
    }
    
    async fn get_conversation(&self, id: &str) -> Result<Option<AgentState>, OpenAIAgentError> {
        let conversations = self.conversations.read().map_err(|e| {
            OpenAIAgentError::Agent(format!("Failed to acquire read lock: {}", e))
        })?;
        
        Ok(conversations.get(id).map(|(state, _)| state.clone()))
    }
    
    async fn delete_conversation(&self, id: &str) -> Result<(), OpenAIAgentError> {
        let mut conversations = self.conversations.write().map_err(|e| {
            OpenAIAgentError::Agent(format!("Failed to acquire write lock: {}", e))
        })?;
        
        conversations.remove(id);
        Ok(())
    }
    
    async fn list_conversations(&self, limit: usize, offset: usize) -> Result<Vec<ConversationMetadata>, OpenAIAgentError> {
        let conversations = self.conversations.read().map_err(|e| {
            OpenAIAgentError::Agent(format!("Failed to acquire read lock: {}", e))
        })?;
        
        let mut metadata: Vec<_> = conversations.values()
            .map(|(_, meta)| meta.clone())
            .collect();
        
        // Sort by updated_at (most recent first)
        metadata.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        
        let end = (offset + limit).min(metadata.len());
        if offset >= metadata.len() {
            return Ok(Vec::new());
        }
        
        Ok(metadata[offset..end].to_vec())
    }
} 