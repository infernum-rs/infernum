//! Chat templates for the `DeepSeek` model family

use infernum::chat_template::{ChatMessage, ChatTemplate};

/// `DeepSeek` V3 / R1 chat template.
///
/// Format:
/// ```text
/// {system content}
///
/// <｜User｜>{content}
/// <｜Assistant｜>{content}<｜end▁of▁sentence｜>
/// <｜User｜>{content}
/// <｜Assistant｜>
/// ```
///
/// Note: The special tokens use fullwidth vertical line (`｜`, U+FF5C) and
/// lower one eighth block (`▁`, U+2581), matching the `DeepSeek` tokenizer.
pub struct DeepSeekTemplate;

impl ChatTemplate for DeepSeekTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str(&msg.content);
                    prompt.push_str(
                        "

",
                    );
                }
                "user" => {
                    prompt.push_str("<｜User｜>");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                "assistant" => {
                    prompt.push_str("<｜Assistant｜>");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<｜end▁of▁sentence｜>");
                }
                _ => {
                    prompt.push_str(&msg.content);
                }
            }
        }
        // Add the assistant header to prompt generation
        prompt.push_str("<｜Assistant｜>");
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_single_user() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = DeepSeekTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            "<｜User｜>Hello
<｜Assistant｜>"
        );
    }

    #[test]
    fn deepseek_system_and_user() {
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
        let prompt = DeepSeekTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            "You are helpful.

<｜User｜>Hi
<｜Assistant｜>"
        );
    }

    #[test]
    fn deepseek_multi_turn() {
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
        let prompt = DeepSeekTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            concat!(
                "<｜User｜>Hello
",
                "<｜Assistant｜>Hi!<｜end\u{2581}of\u{2581}sentence｜>",
                "<｜User｜>How are you?
",
                "<｜Assistant｜>",
            )
        );
    }
}
