//! Chat templates for converting messages to model prompts
//!
//! The [`ChatTemplate`] trait converts a sequence of [`ChatMessage`]s (in the
//! `OpenAI` messages format) into a prompt string suitable for the model.
//!
//! Model crates provide concrete implementations (e.g., `Llama3Template`,
//! `ChatMLTemplate`). The serve crate calls `template.apply(messages)`.

/// A chat message in the `OpenAI` messages format.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role of the message sender (`"system"`, `"user"`, `"assistant"`).
    pub role: String,
    /// Content of the message.
    pub content: String,
}

/// Converts `OpenAI`-style messages into a prompt string for the model.
///
/// Each model family has its own prompt format with special tokens.
/// Implement this trait for each supported format.
pub trait ChatTemplate: Send + Sync {
    /// Apply the template to a list of messages, producing a prompt string.
    fn apply(&self, messages: &[ChatMessage]) -> String;
}

impl ChatTemplate for Box<dyn ChatTemplate> {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        (**self).apply(messages)
    }
}

/// A raw template that concatenates message contents without special tokens.
///
/// Useful for base (non-instruct) models and testing.
pub struct RawTemplate;

impl ChatTemplate for RawTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(
                "
",
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_template_concatenates() {
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Hi there!".into(),
            },
        ];
        let prompt = RawTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            "Hello
Hi there!"
        );
    }
}
