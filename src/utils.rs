//! Utility functions for working with the OpenAI API.
//!
//! This module provides helper functions for common tasks like token counting,
//! text truncation, and implementing retry logic for API requests.

use crate::error::OpenAIAgentError;
use tiktoken_rs::tokenizer::{get_tokenizer, Tokenizer};
use tiktoken_rs::{cl100k_base, p50k_base, r50k_base, p50k_edit, o200k_base, CoreBPE};

/// Gets the appropriate BPE tokenizer for a given OpenAI model name.
///
/// # Arguments
///
/// * `model` - The name of the OpenAI model
///
/// # Returns
///
/// A Result containing either the CoreBPE tokenizer or an error
fn bpe_for_model(model: &str) -> Result<CoreBPE, OpenAIAgentError> {
    // Get the tokenizer type based on the model name
    let tokenizer_enum = get_tokenizer(model)
        .ok_or_else(|| OpenAIAgentError::Parse(format!("Unknown model: {}", model)))?;

    // Initialize the appropriate BPE tokenizer
    let bpe = match tokenizer_enum {
        Tokenizer::Cl100kBase => cl100k_base()
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to init cl100k_base: {}", e)))?,
        Tokenizer::P50kBase => p50k_base()
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to init p50k_base: {}", e)))?,
        Tokenizer::R50kBase => r50k_base()
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to init r50k_base: {}", e)))?,
        Tokenizer::P50kEdit => p50k_edit()
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to init p50k_edit: {}", e)))?,
        Tokenizer::O200kBase => o200k_base()
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to init o200k_base: {}", e)))?,
        Tokenizer::Gpt2 => p50k_base()
            .map_err(|e| OpenAIAgentError::Parse(format!("Failed to init p50k_base for GPT-2: {}", e)))?,
    };

    Ok(bpe)
}

/// Counts the number of tokens in a text string for a given model.
///
/// This function is useful for estimating costs and ensuring that
/// requests stay within token limits.
///
/// # Arguments
///
/// * `text` - The text to count tokens for
/// * `model` - The name of the model to use for tokenization
///
/// # Returns
///
/// A Result containing either the token count or an error
pub fn count_tokens(text: &str, model: &str) -> Result<usize, OpenAIAgentError> {
    let bpe = bpe_for_model(model)?;
    let tokens = bpe.encode_with_special_tokens(text);
    Ok(tokens.len())
}

/// Truncates text to a maximum number of tokens for a given model.
///
/// This function ensures that text stays within token limits by
/// cutting it off at the appropriate point.
///
/// # Arguments
///
/// * `text` - The text to truncate
/// * `max_tokens` - The maximum number of tokens to allow
/// * `model` - The name of the model to use for tokenization
///
/// # Returns
///
/// A Result containing either the truncated text or an error
pub fn truncate_text_to_tokens(text: &str, max_tokens: usize, model: &str) -> Result<String, OpenAIAgentError> {
    let bpe = bpe_for_model(model)?;

    // Encode to tokens
    let tokens = bpe.encode_with_special_tokens(text);

    // Return early if already within limits
    if tokens.len() <= max_tokens {
        return Ok(text.to_string());
    }

    // Truncate tokens
    let truncated_tokens = &tokens[..max_tokens];

    // Decode the truncated token list back to text
    let truncated_text = bpe.decode(Vec::from(truncated_tokens))
        .map_err(|e| OpenAIAgentError::Parse(format!("Failed to decode tokens: {}", e)))?;

    Ok(truncated_text)
}

/// Implements retry logic for API calls with exponential backoff.
///
/// This function will retry a failing operation a specified number of times,
/// with increasing delays between attempts.
///
/// # Arguments
///
/// * `operation` - The async operation to perform
/// * `max_retries` - The maximum number of retry attempts
/// * `initial_delay_ms` - The initial delay in milliseconds before the first retry
///
/// # Returns
///
/// A Result containing either the operation's output or an error after exhausting retries
pub async fn with_retries<F, Fut, T>(
    operation: F,
    max_retries: usize,
    initial_delay_ms: u64,
) -> Result<T, OpenAIAgentError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, OpenAIAgentError>>,
{
    let mut retries = 0;
    let mut delay_ms = initial_delay_ms;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                if retries >= max_retries {
                    return Err(err);
                }

                // Only retry on request errors that might be temporary
                match &err {
                    OpenAIAgentError::Reqwest(req_err) if req_err.is_timeout() || req_err.is_connect() => {
                        // Exponential backoff
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        delay_ms *= 2;
                        retries += 1;
                    },
                    _ => return Err(err),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens() {
        let text = "Hello, world!";
        let result = count_tokens(text, "gpt-3.5-turbo");
        assert!(result.is_ok());
    }

    #[test]
    fn test_truncate_text() {
        let text = "This is a long text that needs to be truncated to fit within token limits.";
        let result = truncate_text_to_tokens(text, 5, "gpt-3.5-turbo");
        assert!(result.is_ok());

        let truncated = result.unwrap();
        assert!(truncated.len() < text.len());
    }
}