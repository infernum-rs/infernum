//! Chat templates for Llama and Mistral model families

use infernum::chat_template::{ChatMessage, ChatTemplate};

/// Llama 3.x instruct chat template.
///
/// Format:
/// ```text
/// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
///
/// {content}<|eot_id|><|start_header_id|>user<|end_header_id|>
///
/// {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
///
///
/// ```
pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::from("<|begin_of_text|>");
        for msg in messages {
            prompt.push_str("<|start_header_id|>");
            prompt.push_str(&msg.role);
            prompt.push_str(
                "<|end_header_id|>

",
            );
            prompt.push_str(&msg.content);
            prompt.push_str("<|eot_id|>");
        }
        // Add the assistant header to prompt generation
        prompt.push_str(
            "<|start_header_id|>assistant<|end_header_id|>

",
        );
        prompt
    }
}

/// Mistral instruct chat template (v1/v2/v3).
///
/// Format:
/// ```text
/// [INST] {system + user content} [/INST]
/// ```
pub struct MistralTemplate;

impl ChatTemplate for MistralTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            if msg.role == "assistant" {
                // Assistant responses are placed after [/INST]
                prompt.push_str(&msg.content);
                prompt.push_str("</s>");
            } else {
                // system and user messages go inside [INST]
                prompt.push_str("[INST] ");
                // If system message, concatenate with following user message
                if msg.role == "system" {
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                    if i + 1 < messages.len() && messages[i + 1].role == "user" {
                        i += 1;
                        prompt.push_str(&messages[i].content);
                    }
                } else {
                    prompt.push_str(&msg.content);
                }
                prompt.push_str(" [/INST]");
            }
            i += 1;
        }
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_single_user() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = Llama3Template.apply(&msgs);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "Hello<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn llama3_system_and_user() {
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
        let prompt = Llama3Template.apply(&msgs);
        assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("You are helpful.<|eot_id|>"));
        assert!(prompt.ends_with(
            "<|start_header_id|>assistant<|end_header_id|>

"
        ));
    }

    #[test]
    fn mistral_single_user() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = MistralTemplate.apply(&msgs);
        assert_eq!(prompt, "[INST] Hello [/INST]");
    }

    #[test]
    fn mistral_system_and_user() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "Be brief.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            },
        ];
        let prompt = MistralTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            "[INST] Be brief.
Hi [/INST]"
        );
    }

    #[test]
    fn mistral_multi_turn() {
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
        let prompt = MistralTemplate.apply(&msgs);
        assert_eq!(
            prompt,
            "[INST] Hello [/INST]Hi!</s>[INST] How are you? [/INST]"
        );
    }
}
