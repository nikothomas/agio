// tests/persistence_test.rs

use std::env;
use std::sync::Arc;
use tokio::runtime::Runtime;

// Import from the public API
use agio::{
    AgentBuilder, Config, Error,
};
// Import persistence and server modules
use agio::persistence::{MemoryStore, PersistenceStore, ConversationMetadata, PostgresStore};
use agio::server::AgentManager;

#[test]
fn test_memory_persistence() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY")
        .expect("Set the OPENAI_API_KEY env var before running this test.");

    // Create a Tokio runtime
    let rt = Runtime::new()?;

    // Run the test
    rt.block_on(async {
        // Create a memory store
        let store = Arc::new(MemoryStore::new());
        
        // Create an agent with persistence
        let mut agent = AgentBuilder::new()
            .with_config(Config::new()
                .with_api_key(api_key.clone())
                .with_model("gpt-3.5-turbo")
                .with_temperature(0.0))
            .with_system_prompt("You are a helpful assistant for testing.")
            .with_persistence(store.clone())
            .build_async()
            .await?;
        
        // Get the agent ID
        let agent_id = agent.id().to_string();
        println!("Created agent with ID: {}", agent_id);
        
        // Run a message
        let response = agent.run("Hello, this is a test message.").await?;
        println!("Response: {}", response);
        
        // Verify the agent was saved
        let conversations = store.list_conversations(10, 0).await?;
        assert!(!conversations.is_empty(), "No conversations found in store");
        
        let found = conversations.iter().any(|c| c.id == agent_id);
        assert!(found, "Agent ID not found in conversations");
        
        // Load the agent by ID
        let loaded_agent = AgentBuilder::new()
            .with_id(agent_id.clone())
            .with_config(Config::new()
                .with_api_key(api_key.clone())
                .with_model("gpt-3.5-turbo")
                .with_temperature(0.0))
            .with_persistence(store.clone())
            .build_async()
            .await?;
        
        // Verify the loaded agent has the same ID
        assert_eq!(loaded_agent.id(), agent_id, "Loaded agent has different ID");
        
        // Verify the loaded agent has the message history
        assert!(loaded_agent.state().message_count() > 0, "Loaded agent has no messages");
        
        // Delete the agent
        loaded_agent.delete().await?;
        
        // Verify the agent was deleted
        let conversations_after_delete = store.list_conversations(10, 0).await?;
        let found_after_delete = conversations_after_delete.iter().any(|c| c.id == agent_id);
        assert!(!found_after_delete, "Agent still exists after deletion");
        
        Ok(())
    })
}

#[test]
fn test_agent_manager() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY")
        .expect("Set the OPENAI_API_KEY env var before running this test.");

    // Create a Tokio runtime
    let rt = Runtime::new()?;

    // Run the test
    rt.block_on(async {
        // Create a memory store
        let store = Arc::new(MemoryStore::new());
        
        // Create an agent manager
        let config = Config::new()
            .with_api_key(api_key)
            .with_model("gpt-3.5-turbo")
            .with_temperature(0.0);
            
        let manager = Arc::new(AgentManager::new(config, store, 10));
        
        // Create a new agent
        let agent_id = manager.create_agent().await?;
        println!("Created agent with ID: {}", agent_id);
        
        // Send a message to the agent
        let response = manager.run_message(&agent_id, "Hello, this is a test message.").await?;
        println!("Response: {}", response);
        
        // List all conversations
        let conversations = manager.list_conversations(10, 0).await?;
        assert!(!conversations.is_empty(), "No conversations found");
        
        let found = conversations.iter().any(|c| c.id == agent_id);
        assert!(found, "Agent ID not found in conversations");
        
        // Create multiple agents to test caching
        let mut agent_ids = vec![agent_id.clone()];
        for i in 0..15 {
            let id = manager.create_agent().await?;
            println!("Created additional agent {}: {}", i, id);
            agent_ids.push(id);
        }
        
        // Verify we can still access the first agent (tests cache eviction)
        let response = manager.run_message(&agent_id, "Testing cache eviction.").await?;
        println!("Response after cache eviction: {}", response);
        
        // Delete an agent
        manager.delete_agent(&agent_id).await?;
        
        // Verify the agent was deleted
        let result = manager.get_agent(&agent_id).await;
        assert!(result.is_err(), "Agent still exists after deletion");
        
        // Clean up all created agents
        for id in agent_ids.iter().skip(1) {
            let _ = manager.delete_agent(id).await;
        }
        
        Ok(())
    })
}

#[test]
fn test_postgres_persistence() -> Result<(), Box<dyn std::error::Error>> {
    // Skip this test if DATABASE_URL is not set
    let db_url = match env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping PostgreSQL test: DATABASE_URL not set");
            return Ok(());
        }
    };
    
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY")
        .expect("Set the OPENAI_API_KEY env var before running this test.");

    // Create a Tokio runtime
    let rt = Runtime::new()?;

    // Run the test
    rt.block_on(async {
        // Create a PostgreSQL store
        let store = match PostgresStore::new(&db_url).await {
            Ok(store) => Arc::new(store),
            Err(e) => {
                println!("Skipping PostgreSQL test: Failed to connect to database: {}", e);
                return Ok(());
            }
        };
        
        // Create an agent with persistence
        let mut agent = AgentBuilder::new()
            .with_config(Config::new()
                .with_api_key(api_key.clone())
                .with_model("gpt-3.5-turbo")
                .with_temperature(0.0))
            .with_system_prompt("You are a helpful assistant for testing PostgreSQL persistence.")
            .with_persistence(store.clone())
            .build_async()
            .await?;
        
        // Get the agent ID
        let agent_id = agent.id().to_string();
        println!("Created agent with ID: {}", agent_id);
        
        // Run a message
        let response = agent.run("Hello, this is a PostgreSQL test message.").await?;
        println!("Response: {}", response);
        
        // Verify the agent was saved
        let conversations = store.list_conversations(10, 0).await?;
        assert!(!conversations.is_empty(), "No conversations found in PostgreSQL store");
        
        let found = conversations.iter().any(|c| c.id == agent_id);
        assert!(found, "Agent ID not found in PostgreSQL conversations");
        
        // Load the agent by ID
        let loaded_agent = AgentBuilder::new()
            .with_id(agent_id.clone())
            .with_config(Config::new()
                .with_api_key(api_key.clone())
                .with_model("gpt-3.5-turbo")
                .with_temperature(0.0))
            .with_persistence(store.clone())
            .build_async()
            .await?;
        
        // Verify the loaded agent has the same ID
        assert_eq!(loaded_agent.id(), agent_id, "Loaded agent has different ID");
        
        // Verify the loaded agent has the message history
        assert!(loaded_agent.state().message_count() > 0, "Loaded agent has no messages");
        
        // Delete the agent
        loaded_agent.delete().await?;
        
        // Verify the agent was deleted
        let conversations_after_delete = store.list_conversations(10, 0).await?;
        let found_after_delete = conversations_after_delete.iter().any(|c| c.id == agent_id);
        assert!(!found_after_delete, "Agent still exists after deletion in PostgreSQL");
        
        Ok(())
    })
} 