//! Example web server using the persistence and server functionality.
//!
//! This example demonstrates how to use the AgentManager to create a simple
//! web server that manages multiple agents with persistence.
//!
//! To run this example:
//! ```
//! cargo run --example web_server
//! ```

use agio::{
    Config, 
    persistence::MemoryStore,
    server::AgentManager,
};
use axum::{
    routing::{get, post, delete},
    Router, Json, extract::{State, Path},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::net::SocketAddr;

#[derive(Deserialize)]
struct MessageRequest {
    message: String,
}

#[derive(Serialize)]
struct MessageResponse {
    response: String,
}

#[derive(Serialize)]
struct AgentResponse {
    id: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
    
    // Setup persistence (using in-memory store for this example)
    let store = Arc::new(MemoryStore::new());
    
    // Create agent manager with default configuration
    let config = Config::new()
        .with_api_key(api_key)
        .with_model("gpt-4o");
        
    let agent_manager = Arc::new(AgentManager::new(config, store, 100));
    
    // Setup API routes
    let app = Router::new()
        .route("/agents", post(create_agent))
        .route("/agents/{id}/messages", post(handle_message))
        .route("/agents/{id}", get(get_agent))
        .route("/agents/{id}", delete(delete_agent))
        .route("/agents", get(list_agents))
        .with_state(agent_manager);
        
    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Starting server on http://{}", addr);
    
    // Use the correct axum server binding API
    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app
    )
    .await?;
        
    Ok(())
}

/// Create a new agent
async fn create_agent(
    State(manager): State<Arc<AgentManager>>,
) -> Result<Json<AgentResponse>, String> {
    manager.create_agent().await
        .map(|id| Json(AgentResponse { id }))
        .map_err(|e| e.to_string())
}

/// Send a message to an agent
async fn handle_message(
    State(manager): State<Arc<AgentManager>>,
    Path(id): Path<String>,
    Json(request): Json<MessageRequest>,
) -> Result<Json<MessageResponse>, String> {
    manager.run_message(&id, &request.message).await
        .map(|response| Json(MessageResponse { response }))
        .map_err(|e| e.to_string())
}

/// Check if an agent exists
async fn get_agent(
    State(manager): State<Arc<AgentManager>>,
    Path(id): Path<String>,
) -> Result<String, String> {
    manager.get_agent(&id).await
        .map(|_| format!("Agent {} exists", id))
        .map_err(|e| e.to_string())
}

/// Delete an agent
async fn delete_agent(
    State(manager): State<Arc<AgentManager>>,
    Path(id): Path<String>,
) -> Result<String, String> {
    manager.delete_agent(&id).await
        .map(|_| format!("Agent {} deleted", id))
        .map_err(|e| e.to_string())
}

/// List all agents
async fn list_agents(
    State(manager): State<Arc<AgentManager>>,
) -> Result<Json<Vec<agio::persistence::ConversationMetadata>>, String> {
    manager.list_conversations(100, 0).await
        .map(Json)
        .map_err(|e| e.to_string())
} 