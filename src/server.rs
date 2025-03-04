//! Server-side components for managing agents in a server environment.
//!
//! This module provides utilities for managing multiple agents in a server context,
//! including caching, eviction, and database persistence.

use crate::agent::{Agent, AgentBuilder};
use crate::config::OpenAIConfig;
use crate::error::OpenAIAgentError;
use crate::persistence::{EntityId, PersistenceStore, ConversationMetadata};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Server-side agent manager for handling multiple concurrent agents
pub struct AgentManager {
    /// Default configuration for new agents
    config: OpenAIConfig,
    
    /// Persistence store for agent data
    store: Arc<dyn PersistenceStore>,
    
    /// Cache of active agents
    active_agents: RwLock<HashMap<EntityId, Arc<RwLock<Agent>>>>,
    
    /// Maximum number of agents to keep in memory
    max_cached_agents: usize,
}

impl AgentManager {
    /// Create a new agent manager
    pub fn new(config: OpenAIConfig, store: Arc<dyn PersistenceStore>, max_cached_agents: usize) -> Self {
        Self {
            config,
            store,
            active_agents: RwLock::new(HashMap::new()),
            max_cached_agents,
        }
    }
    
    /// Create a new agent
    pub async fn create_agent(&self) -> Result<EntityId, OpenAIAgentError> {
        let agent = AgentBuilder::new()
            .with_config(self.config.clone())
            .with_persistence(self.store.clone())
            .build_async()
            .await?;
            
        let id = agent.id().to_string();
        let agent = Arc::new(RwLock::new(agent));
        
        // Add to cache
        {
            let mut agents = self.active_agents.write().await;
            agents.insert(id.clone(), agent);
            self.evict_if_needed(&mut agents).await;
        }
        
        Ok(id)
    }
    
    /// Get an existing agent by ID
    pub async fn get_agent(&self, id: &str) -> Result<Arc<RwLock<Agent>>, OpenAIAgentError> {
        // Check if agent is already in memory
        {
            let agents = self.active_agents.read().await;
            if let Some(agent) = agents.get(id) {
                return Ok(agent.clone());
            }
        }
        
        // Load agent from persistence
        let agent = AgentBuilder::new()
            .with_id(id)
            .with_config(self.config.clone())
            .with_persistence(self.store.clone())
            .build_async()
            .await?;
            
        // Check if agent was actually loaded
        if agent.state().message_count() == 0 {
            return Err(OpenAIAgentError::Agent(format!("Agent not found: {}", id)));
        }
        
        let agent = Arc::new(RwLock::new(agent));
        
        // Add to cache
        {
            let mut agents = self.active_agents.write().await;
            agents.insert(id.to_string(), agent.clone());
            self.evict_if_needed(&mut agents).await;
        }
        
        Ok(agent)
    }
    
    /// Run a message through an agent
    pub async fn run_message(&self, agent_id: &str, message: &str) -> Result<String, OpenAIAgentError> {
        let agent_lock = self.get_agent(agent_id).await?;
        let mut agent = agent_lock.write().await;
        agent.run(message).await
    }
    
    /// Delete an agent and its data
    pub async fn delete_agent(&self, id: &str) -> Result<(), OpenAIAgentError> {
        // Remove from cache
        {
            let mut agents = self.active_agents.write().await;
            agents.remove(id);
        }
        
        // Delete from storage
        self.store.delete_conversation(id).await
    }
    
    /// List available conversations
    pub async fn list_conversations(&self, limit: usize, offset: usize) -> Result<Vec<ConversationMetadata>, OpenAIAgentError> {
        self.store.list_conversations(limit, offset).await
    }
    
    /// Evict agents from cache if needed
    async fn evict_if_needed(&self, agents: &mut HashMap<EntityId, Arc<RwLock<Agent>>>) {
        if agents.len() <= self.max_cached_agents {
            return;
        }
        
        // Simple LRU-like eviction - remove oldest entries first
        // In a real implementation, you'd want to track last access time
        if agents.len() > self.max_cached_agents {
            let keys: Vec<_> = agents.keys().cloned().collect();
            let to_remove = keys.len() - self.max_cached_agents;
            
            for key in keys.into_iter().take(to_remove) {
                agents.remove(&key);
            }
        }
    }
} 