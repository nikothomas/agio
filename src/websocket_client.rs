//! WebSocket client for the OpenAI Realtime API (Beta).
//!
//! This module provides functionality for establishing a WebSocket connection
//! to the new OpenAI Realtime API. See:
//! https://api.openai.com/v1/realtime
//! for more details on the Beta.

use base64::Engine;
use base64::engine::general_purpose;
use crate::config::OpenAIConfig;
use crate::error::OpenAIAgentError;
use futures_util::{SinkExt, StreamExt};
use rand::RngCore;
// Bring in Rustls so we can check the crypto provider
use rustls::crypto::CryptoProvider;

use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tokio_tungstenite::tungstenite::{handshake::client::Request, Message};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use url::Url;

/// Example of an outbound event you might send to the Realtime API.
/// Adjust fields based on the official Realtime specs.
#[derive(Serialize, Deserialize, Debug)]
pub struct RealtimeEvent {
    pub r#type: String,
    // Add any additional fields your event needs...
}

/// Example of an inbound server event from the Realtime API.
/// Adjust fields based on the official Realtime specs.
#[derive(Serialize, Deserialize, Debug)]
pub struct ServerEvent {
    pub event_type: String,
    // Add more fields matching your Realtime server messages...
}

/// Handler for an OpenAI Realtime WebSocket connection.
pub struct WebSocketClient {
    /// Configuration that holds your API key, base URL, etc.
    config: OpenAIConfig,

    /// The underlying WebSocket connection, if established.
    connection: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
}

impl WebSocketClient {
    /// Creates a new WebSocketClient with the given config.
    pub fn new(config: OpenAIConfig) -> Result<Self, OpenAIAgentError> {
        CryptoProvider::get_default();
        if config.api_key().is_empty() {
            return Err(OpenAIAgentError::Config(
                "API key not provided".to_string(),
            ));
        }

        Ok(Self {
            config,
            connection: None,
        })
    }

    pub async fn connect(&mut self, model_name: &str) -> Result<(), OpenAIAgentError> {
        // Convert the base URL from https->wss, etc., removing any trailing slash.
        let base_url = self.config.base_url();
        let ws_base = base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            .trim_end_matches('/')
            .to_string();

        // Append "/realtime" so you end with something like "wss://api.openai.com/v1/realtime"
        let realtime_path = format!("{}/realtime", ws_base);

        // Build the URL with the "model" query parameter.
        let url = Url::parse_with_params(&realtime_path, &[("model", model_name)])
            .map_err(|e| OpenAIAgentError::Config(format!("Invalid WebSocket URL: {}", e)))?;

        println!("Connecting to Realtime API... {}", url);

        // Generate a valid Sec-WebSocket-Key.
        let mut key = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut key);
        let key_base64 = general_purpose::STANDARD.encode(&key);

        // Extract the host from the URL to use in the header.
        let host = url.host_str().ok_or_else(|| {
            OpenAIAgentError::Config("URL missing host information".to_string())
        })?;

        // Convert the URL string into a client request.
        let url_str = url.to_string();
        let mut request = url_str.into_client_request().map_err(|e| {
            OpenAIAgentError::Config(format!("Request conversion error: {}", e))
        })?;

        // Insert the required headers.
        {
            let headers = request.headers_mut();
            headers.insert("Host", host.parse().unwrap());
            headers.insert("Authorization", format!("Bearer {}", self.config.api_key()).parse().unwrap());
            headers.insert("OpenAI-Beta", "realtime=v1".parse().unwrap());
            headers.insert("Sec-WebSocket-Key", key_base64.parse().unwrap());
            headers.insert("Sec-WebSocket-Version", "13".parse().unwrap());
        }

        // Perform the async WebSocket handshake.
        let (ws_stream, response) = connect_async(request)
            .await
            .map_err(|e| OpenAIAgentError::Request(format!("WebSocket connection failed: {}", e)))?;

        println!("Connected to Realtime API with HTTP status: {}", response.status());
        self.connection = Some(ws_stream);
        Ok(())
    }




    /// Sends a JSON-encoded Realtime event to the API.
    ///
    /// Adjust this as needed to match the official Realtime event schema.
    pub async fn send_event(&mut self, event: &RealtimeEvent) -> Result<(), OpenAIAgentError> {
        let connection = match &mut self.connection {
            Some(conn) => conn,
            None => return Err(OpenAIAgentError::Request("Not connected".to_string())),
        };

        let payload = serde_json::to_string(event)
            .map_err(OpenAIAgentError::Serialization)?;
        connection
            .send(Message::Text(payload.into()))
            .await
            .map_err(|e| OpenAIAgentError::Request(format!("Failed to send event: {}", e)))?;

        Ok(())
    }

    /// Reads incoming messages from the Realtime API in a loop,
    /// calling the provided handler function for each one.
    ///
    /// This will run until the server closes the connection or an error occurs.
    pub async fn process_incoming<F>(&mut self, mut on_event: F) -> Result<(), OpenAIAgentError>
    where
    // Handler: given a parsed ServerEvent, return a result or fail
        F: FnMut(ServerEvent) -> Result<(), OpenAIAgentError>,
    {
        let connection = match &mut self.connection {
            Some(conn) => conn,
            None => return Err(OpenAIAgentError::Request("Not connected".to_string())),
        };

        while let Some(msg_result) = connection.next().await {
            let msg = msg_result
                .map_err(|e| OpenAIAgentError::Request(format!("WebSocket read error: {}", e)))?;

            match msg {
                Message::Text(txt) => {
                    // Typically, the server event is JSON. Let's parse it:
                    match serde_json::from_str::<ServerEvent>(&txt) {
                        Ok(server_event) => {
                            on_event(server_event)?;
                        }
                        Err(parse_err) => {
                            eprintln!("Failed to parse inbound JSON: {}", parse_err);
                            // Depending on your logic, you might skip or return an error here
                        }
                    }
                }
                Message::Binary(bin) => {
                    // If Realtime API ever sends binary data, handle it here. Possibly audio or other media.
                    println!("Got {} bytes of binary data", bin.len());
                }
                Message::Close(frame) => {
                    println!("Server closed the connection: {:?}", frame);
                    break; // Exit the loop
                }
                _ => {
                    // Ping/Pong or other messages—handle if desired
                }
            }
        }

        Ok(())
    }

    /// Closes the WebSocket connection gracefully.
    pub async fn close(&mut self) -> Result<(), OpenAIAgentError> {
        if let Some(conn) = &mut self.connection {
            conn.close(None)
                .await
                .map_err(|e| OpenAIAgentError::Request(format!("Failed to close connection: {}", e)))?;
            self.connection = None;
        }
        Ok(())
    }
}

impl Drop for WebSocketClient {
    fn drop(&mut self) {
        // Attempt to close on drop if still connected
        if self.connection.is_some() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.block_on(async {
                    let _ = self.close().await;
                });
            }
        }
    }
}
