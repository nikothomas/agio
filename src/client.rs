//! HTTP client implementation for communicating with the OpenAI API.
//!
//! This module provides the `OpenAIClient` which handles the HTTP communication
//! with OpenAI's API, including authentication, request formatting, and response parsing.

use crate::error::OpenAIAgentError;
use crate::models::{ChatRequest, ChatResponse};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use crate::Config;

/// Client for making requests to the OpenAI API.
///
/// This struct handles HTTP communications with the OpenAI API,
/// including authentication and request/response formatting.
pub struct OpenAIClient {
    /// Configuration for the OpenAI API
    config: Config,

    /// HTTP client for making requests
    pub client: reqwest::Client,
}

impl OpenAIClient {
    /// Creates a new OpenAI client with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to use for API requests
    ///
    /// # Returns
    ///
    /// A Result containing either the constructed client or an error
    pub fn new(config: Config) -> Result<Self, OpenAIAgentError> {
        if config.api_key().is_empty() {
            return Err(OpenAIAgentError::Config("API key not provided".to_string()));
        }

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", config.api_key()))
                .map_err(|_| OpenAIAgentError::Config("Invalid API key format".to_string()))?,
        );

        if let Some(org) = &config.organization() {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org)
                    .map_err(|_| OpenAIAgentError::Config("Invalid organization ID format".to_string()))?,
            );
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(config.timeout())
            .build()
            .map_err(|e| OpenAIAgentError::Request(e.to_string()))?;

        Ok(Self { config, client })
    }

    /// Sends a chat completion request to the OpenAI API.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat request to send
    ///
    /// # Returns
    ///
    /// A Result containing either the API response or an error
    pub async fn chat_completion(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, OpenAIAgentError> {
        let url = format!("{}/chat/completions", self.config.base_url());
        let response = self.client.post(&url).json(&request).send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(OpenAIAgentError::Request(format!(
                "HTTP error {}: {}",
                status, error_text
            )));
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response)
    }

    /// Returns a reference to the client's configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ChatMessage;
    use mockito;

    #[test]
    fn test_chat_completion() {
        let mut mock_server = mockito::Server::new();
        let mock_response = r#"{
            "id": "test-id",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        }"#;

        let _mock = mock_server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_response)
            .create();

        let config = Config::new()
            .with_api_key("test-api-key")
            .with_base_url(&mock_server.url())
            .with_timeout(std::time::Duration::from_secs(10));

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let client = OpenAIClient::new(config).unwrap();
            let request = ChatRequest {
                model: "gpt-3.5-turbo".to_string(),
                messages: vec![ChatMessage::user("Hello!")],
                max_tokens: None,
                temperature: None,
                response_format: None,
                stream: None,
                tools: None,
            };

            let response = client.chat_completion(request).await;
            assert!(response.is_ok());
            let response = response.unwrap();
            assert_eq!(response.choices.len(), 1);
            let choice_msg = &response.choices[0].message;
            assert_eq!(choice_msg.content.as_ref().unwrap(), "Hello! How can I help you today?");
        });
    }
}