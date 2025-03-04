//! Configuration options for the OpenAI API client.
//!
//! This module provides a configuration struct for customizing the behavior
//! of the OpenAI API client, including API keys, model selection, and request parameters.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the OpenAI API client.
///
/// This struct contains all the settings needed to customize requests to OpenAI,
/// including authentication, model selection, and request parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// OpenAI API key for authentication
    api_key: String,

    /// Model identifier to use (e.g., "gpt-4o", "gpt-4", "gpt-3.5-turbo")
    model: String,

    /// Base URL for the OpenAI API
    #[serde(default = "default_base_url")]
    base_url: String,

    /// Organization ID for team accounts (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    organization: Option<String>,

    /// Timeout duration for API requests
    #[serde(with = "humantime_serde", default = "default_timeout")]
    timeout: Duration,

    /// Maximum number of tokens to generate in responses
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    /// Temperature setting for response randomness (0.0 to 2.0)
    #[serde(default = "default_temperature")]
    temperature: f32,

    /// Whether to enable JSON mode for structured outputs
    #[serde(default)]
    json_mode: bool,

    /// Whether to stream responses instead of waiting for completion
    #[serde(default)]
    stream: bool,
}

/// Default base URL for the OpenAI API.
fn default_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

/// Default timeout duration for API requests.
fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

/// Default maximum number of tokens to generate.
fn default_max_tokens() -> usize {
    1024
}

/// Default temperature setting for response randomness.
fn default_temperature() -> f32 {
    0.7
}

impl OpenAIConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self {
            api_key: String::new(),
            model: "gpt-4".to_string(),
            base_url: default_base_url(),
            organization: None,
            timeout: default_timeout(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            json_mode: false,
            stream: false,
        }
    }

    /// Sets the API key for authentication.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The OpenAI API key
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Sets the model to use for requests.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the base URL for the API.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL for API requests
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the organization ID for team accounts.
    ///
    /// # Arguments
    ///
    /// * `organization` - The organization identifier
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Sets the timeout duration for API requests.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The timeout duration
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the maximum number of tokens to generate.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - The maximum token count
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the temperature for response randomness.
    ///
    /// # Arguments
    ///
    /// * `temperature` - The temperature value (0.0 to 2.0)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Enables or disables JSON mode for structured outputs.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable JSON mode
    pub fn with_json_mode(mut self, enabled: bool) -> Self {
        self.json_mode = enabled;
        self
    }

    /// Enables or disables streaming responses.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable streaming
    pub fn with_stream(mut self, enabled: bool) -> Self {
        self.stream = enabled;
        self
    }

    /// Returns the API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the organization ID if set.
    pub fn organization(&self) -> Option<&str> {
        self.organization.as_deref()
    }

    /// Returns the timeout duration.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the maximum token count.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns the temperature setting.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Returns whether JSON mode is enabled.
    pub fn json_mode(&self) -> bool {
        self.json_mode
    }

    /// Returns whether streaming is enabled.
    pub fn stream(&self) -> bool {
        self.stream
    }
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self::new()
    }
}