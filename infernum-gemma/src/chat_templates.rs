//! Chat templates for the Gemma model family

use infernum::chat_template::{ChatMessage, ChatTemplate};

/// Gemma instruct chat template (Gemma 2 and Gemma 3).
///
/// Format:
/// ```text
/// <start_of_turn>user
/// {content}<end_of_turn>
/// <start_of_turn>model
/// {content}<end_of_turn>
/// <start_of_turn>model
///
/// ```
pub struct GemmaTemplate;

impl ChatTemplate for GemmaTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str("<start_of_turn>");
            // Gemma uses "model" instead of "assistant"
            let role = if msg.role == "assistant" {
                "model"
            } else {
                &msg.role
            };
            prompt.push_str(role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str(
                "<end_of_turn>
",
            );
        }
        // Add the model header to prompt generation
        prompt.push_str(
            "<start_of_turn>model
",
        );
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_single_user() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = GemmaTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            "<start_of_turn>user
Hello<end_of_turn>
<start_of_turn>model
"
        );
    }

    #[test]
    fn gemma_system_and_user() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            },
        ];
        let prompt = GemmaTemplate.apply(&msgs);
        let expected = concat!(
            "<start_of_turn>system
You are helpful.<end_of_turn>
",
            "<start_of_turn>user
Hi<end_of_turn>
",
            "<start_of_turn>model
",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn gemma_multi_turn() {
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Hi!".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "How are you?".into(),
            },
        ];
        let prompt = GemmaTemplate.apply(&msgs);
        assert!(prompt.contains(
            "<start_of_turn>model
Hi!<end_of_turn>"
        ));
        assert!(prompt.ends_with(
            "<start_of_turn>model
"
        ));
    }

    #[test]
    fn gemma_assistant_mapped_to_model() {
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Hello!".into(),
            },
        ];
        let prompt = GemmaTemplate.apply(&msgs);
        assert!(!prompt.contains("assistant"));
        assert!(prompt.contains(
            "<start_of_turn>model
Hello!"
        ));
    }
}
