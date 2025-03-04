//! PostgreSQL implementation of the persistence store.
//!
//! This module provides a PostgreSQL-based implementation of the PersistenceStore
//! trait for production use.

use super::{ConversationMetadata, EntityId, PersistenceStore};
use crate::agent::AgentState;
use crate::error::OpenAIAgentError;
use crate::models::ChatMessage;
use async_trait::async_trait;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};

/// PostgreSQL implementation of PersistenceStore
pub struct PostgresStore {
    pool: PgPool,
}

impl PostgresStore {
    /// Create a new PostgreSQL store with the given connection string
    pub async fn new(connection_string: &str) -> Result<Self, OpenAIAgentError> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(connection_string)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Database connection error: {}", e)))?;
            
        // Initialize tables
        Self::init_tables(&pool).await?;
        
        Ok(Self { pool })
    }
    
    async fn init_tables(pool: &PgPool) -> Result<(), OpenAIAgentError> {
        // Create tables if they don't exist - split into separate queries
        println!("Creating conversations table...");
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                message_count INTEGER NOT NULL DEFAULT 0,
                token_count INTEGER NOT NULL DEFAULT 0
            )
            "#
        )
        .execute(pool)
        .await
        .map_err(|e| OpenAIAgentError::Agent(format!("Failed to create conversations table: {}", e)))?;
        
        println!("Creating messages table...");
        // Ensure the messages table is created with the correct column name
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT,
                name TEXT,
                tool_call_id TEXT,
                tool_calls JSONB,
                position INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            "#
        )
        .execute(pool)
        .await
        .map_err(|e| OpenAIAgentError::Agent(format!("Failed to create messages table: {}", e)))?;
        
        // Verify the messages table exists and has the conversation_id column
        println!("Verifying messages table structure...");
        let table_check = sqlx::query("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'messages')")
            .fetch_one(pool)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to check if messages table exists: {}", e)))?;
            
        let table_exists: bool = table_check.get(0);
        if !table_exists {
            return Err(OpenAIAgentError::Agent("Messages table was not created properly".to_string()));
        }
        
        let column_check = sqlx::query("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'messages' AND column_name = 'conversation_id')")
            .fetch_one(pool)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to check if conversation_id column exists: {}", e)))?;
            
        let column_exists: bool = column_check.get(0);
        if !column_exists {
            return Err(OpenAIAgentError::Agent("conversation_id column does not exist in messages table".to_string()));
        }
        
        println!("Creating index on messages.conversation_id...");
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)
            "#
        )
        .execute(pool)
        .await
        .map_err(|e| OpenAIAgentError::Agent(format!("Failed to create index: {}", e)))?;
        
        println!("Database initialization complete");
        Ok(())
    }
}

#[async_trait]
impl PersistenceStore for PostgresStore {
    async fn store_conversation(&self, id: &str, state: &AgentState) -> Result<(), OpenAIAgentError> {
        // Start a transaction
        let mut tx = self.pool.begin().await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to start transaction: {}", e)))?;
        
        // Insert or update conversation metadata
        sqlx::query(
            r#"
            INSERT INTO conversations (id, message_count, token_count, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (id) DO UPDATE SET
                message_count = $2,
                token_count = $3,
                updated_at = NOW()
            "#
        )
        .bind(id)
        .bind(state.message_count() as i32)
        .bind(state.token_count() as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| OpenAIAgentError::Agent(format!("Failed to update conversation: {}", e)))?;
        
        // Delete existing messages
        sqlx::query("DELETE FROM messages WHERE conversation_id = $1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to delete messages: {}", e)))?;
        
        // Insert new messages
        for (i, message) in state.messages().enumerate() {
            let tool_calls_json = if let Some(tool_calls) = &message.tool_calls {
                serde_json::to_value(tool_calls)
                    .map_err(|e| OpenAIAgentError::Agent(format!("Failed to serialize tool calls: {}", e)))?
            } else {
                serde_json::Value::Null
            };
            
            sqlx::query(
                r#"
                INSERT INTO messages (
                    id, conversation_id, role, content, name, 
                    tool_call_id, tool_calls, position, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                "#
            )
            .bind(format!("{}-msg-{}", id, i))
            .bind(id)
            .bind(&message.role)
            .bind(&message.content)
            .bind(&message.name)
            .bind(&message.tool_call_id)
            .bind(tool_calls_json)
            .bind(i as i32)
            .execute(&mut *tx)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to insert message: {}", e)))?;
        }
        
        // Commit transaction
        tx.commit().await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to commit transaction: {}", e)))?;
        
        Ok(())
    }
    
    async fn get_conversation(&self, id: &str) -> Result<Option<AgentState>, OpenAIAgentError> {
        // Check if conversation exists
        let exists = sqlx::query("SELECT 1 FROM conversations WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Database error: {}", e)))?;
            
        if exists.is_none() {
            return Ok(None);
        }
        
        // Get token count
        let token_count: i32 = sqlx::query_scalar("SELECT token_count FROM conversations WHERE id = $1")
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to get token count: {}", e)))?;
        
        // Get messages
        let rows = sqlx::query(
            r#"
            SELECT role, content, name, tool_call_id, tool_calls
            FROM messages
            WHERE conversation_id = $1
            ORDER BY position ASC
            "#
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| OpenAIAgentError::Agent(format!("Failed to fetch messages: {}", e)))?;
        
        let mut messages = Vec::with_capacity(rows.len());
        
        for row in rows {
            let role: String = row.get("role");
            let content: Option<String> = row.get("content");
            let name: Option<String> = row.get("name");
            let tool_call_id: Option<String> = row.get("tool_call_id");
            let tool_calls_json: Option<serde_json::Value> = row.get("tool_calls");
            
            let tool_calls = if let Some(json) = tool_calls_json {
                if json.is_null() {
                    None
                } else {
                    Some(serde_json::from_value(json)
                        .map_err(|e| OpenAIAgentError::Deserialization(e.to_string()))?)
                }
            } else {
                None
            };
            
            let message = ChatMessage {
                role,
                content,
                name,
                tool_call_id,
                tool_calls,
            };
            
            messages.push(message);
        }
        
        let state = AgentState {
            messages,
            token_count: token_count as usize,
        };
        
        Ok(Some(state))
    }
    
    async fn delete_conversation(&self, id: &str) -> Result<(), OpenAIAgentError> {
        // The messages will be deleted automatically due to the ON DELETE CASCADE constraint
        sqlx::query("DELETE FROM conversations WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| OpenAIAgentError::Agent(format!("Failed to delete conversation: {}", e)))?;
            
        Ok(())
    }
    
    async fn list_conversations(&self, limit: usize, offset: usize) -> Result<Vec<ConversationMetadata>, OpenAIAgentError> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, created_at, updated_at, message_count, token_count
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT $1 OFFSET $2
            "#
        )
        .bind(limit as i64)
        .bind(offset as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| OpenAIAgentError::Agent(format!("Failed to list conversations: {}", e)))?;
        
        let mut metadata = Vec::with_capacity(rows.len());
        
        for row in rows {
            let meta = ConversationMetadata {
                id: row.get("id"),
                name: row.get("name"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                message_count: row.get::<i32, _>("message_count") as usize,
                token_count: row.get::<i32, _>("token_count") as usize,
            };
            
            metadata.push(meta);
        }
        
        Ok(metadata)
    }
} 