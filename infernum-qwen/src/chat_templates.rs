//! Chat templates for the Qwen model family

use infernum::chat_template::{ChatMessage, ChatTemplate};

/// `ChatML` template used by Qwen2/2.5/3 models.
///
/// Format:
/// ```text
/// <|im_start|>system
/// {content}<|im_end|>
/// <|im_start|>user
/// {content}<|im_end|>
/// <|im_start|>assistant
///
/// ```
pub struct ChatMLTemplate;

impl ChatTemplate for ChatMLTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str(
                "<|im_end|>
",
            );
        }
        // Add the assistant header to prompt generation
        prompt.push_str(
            "<|im_start|>assistant
",
        );
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_single_user() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = ChatMLTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            "<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
"
        );
    }

    #[test]
    fn chatml_system_and_user() {
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
        let prompt = ChatMLTemplate.apply(&msgs);
        let expected = concat!(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n",
            "<|im_start|>user\nHi<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn chatml_multi_turn() {
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
        let prompt = ChatMLTemplate.apply(&msgs);
        assert!(prompt.contains("Hi!<|im_end|>"));
        assert!(prompt.ends_with(
            "<|im_start|>assistant
"
        ));
    }
}
