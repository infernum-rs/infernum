//! Integration tests for the infernum-serve HTTP server.
//!
//! Uses a mock model and tokenizer to test the API endpoints without
//! requiring a real model download.

#![cfg(feature = "cuda")]

use std::net::SocketAddr;

use infernum::chat_template::RawTemplate;
use infernum::BlockTable;
use infernum::{ModelConfig, Result as InfernumResult, Tokenizer};
use infernum_cuda::cuda::{CudaContext, CudaTensor, PagedKvCache};
use infernum_cuda::{CudaBackend, CudaLogits, CudaRuntimeState};
use infernum_serve::{ModelEntry, Server};

// ---------------------------------------------------------------------------
// MockTokenizer — deterministic, no model files needed
// ---------------------------------------------------------------------------

struct MockTokenizer;

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str, _add_bos: bool) -> InfernumResult<Vec<u32>> {
        // Simple: one token per word, IDs are just word indices + 1
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| u32::try_from(i + 1).unwrap())
            .collect())
    }

    fn decode(&self, ids: &[u32]) -> InfernumResult<String> {
        Ok(ids
            .iter()
            .map(|id| format!("tok{id}"))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn decode_token(&self, id: u32) -> InfernumResult<String> {
        Ok(format!("tok{id}"))
    }

    fn eos_token_id(&self) -> u32 {
        99
    }
}

// ---------------------------------------------------------------------------
// MockModel — generates fixed tokens, no real weights
// ---------------------------------------------------------------------------

struct MockModel {
    ctx: CudaContext,
    vocab_size: usize,
}

impl MockModel {
    fn new(ctx: &CudaContext) -> Self {
        Self {
            ctx: ctx.clone(),
            vocab_size: 100,
        }
    }

    fn make_logits_tensor(&self, batch_size: usize, next_token: u32) -> InfernumResult<CudaTensor> {
        let mut logits = vec![0.0_f32; batch_size * self.vocab_size];
        for b in 0..batch_size {
            logits[b * self.vocab_size + next_token as usize] = 100.0;
        }
        CudaTensor::from_slice(&self.ctx, &[batch_size, self.vocab_size], &logits)
    }
}

impl infernum::Model for MockModel {
    type B = CudaBackend;
    type KvCache = PagedKvCache;

    fn config(&self) -> ModelConfig {
        ModelConfig {
            num_layers: 1,
            max_seq_len: 128,
            num_kv_heads: 1,
            head_dim: 4,
            eos_token_id: 99,
            cache_dtype: infernum::DType::F32,
        }
    }

    fn allocate_kv_cache(
        &self,
        block_config: &infernum::BlockConfig,
    ) -> InfernumResult<Self::KvCache> {
        let mc = self.config();
        PagedKvCache::new(
            &self.ctx,
            mc.num_layers,
            block_config,
            mc.num_kv_heads,
            mc.head_dim,
            mc.cache_dtype,
        )
    }

    fn forward(&self, _input_ids: &[u32]) -> InfernumResult<CudaLogits> {
        Ok(CudaLogits::new(self.make_logits_tensor(1, 42)?))
    }

    fn forward_prefill(
        &self,
        _input_ids: &[u32],
        _kv_cache: &mut Self::KvCache,
        _runtime_state: &mut CudaRuntimeState,
        _block_table: &BlockTable,
        _start_pos: usize,
    ) -> InfernumResult<CudaLogits> {
        Ok(CudaLogits::new(self.make_logits_tensor(1, 42)?))
    }

    fn device(&self) -> &CudaContext {
        &self.ctx
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        _token_ids: &CudaTensor,
        _kv_cache: &mut Self::KvCache,
        _runtime_state: &mut CudaRuntimeState,
        _block_tables: &CudaTensor,
        _seq_lens: &CudaTensor,
        _positions: &CudaTensor,
        batch_size: usize,
        _max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> InfernumResult<CudaLogits> {
        Ok(CudaLogits::new(self.make_logits_tensor(batch_size, 42)?))
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

async fn spawn_test_server() -> SocketAddr {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = MockModel::new(&ctx);
    let entry = ModelEntry::new("test-model", model, MockTokenizer, RawTemplate);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");

    let server = Server::builder().add_model(entry).build();

    tokio::spawn(async move {
        server.run_on(listener).await.expect("server run");
    });

    // Give server a moment to start
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    addr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_list_models() {
    let addr = spawn_test_server().await;
    let url = format!("http://{addr}/v1/models");

    let resp = reqwest::get(&url).await.expect("GET /v1/models");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "list");

    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "test-model");
}

#[tokio::test]
async fn test_non_streaming_completion() {
    let addr = spawn_test_server().await;
    let url = format!("http://{addr}/v1/chat/completions");

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 3
    });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await.expect("POST");
    assert_eq!(resp.status(), 200);

    let result: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(result["object"], "chat.completion");
    assert_eq!(result["model"], "test-model");

    let choices = result["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0]["message"]["role"], "assistant");

    // Should have generated some content
    let content = choices[0]["message"]["content"].as_str().expect("content");
    assert!(!content.is_empty(), "Expected non-empty content");

    // Usage should be present
    assert!(result["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(result["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_model_not_found() {
    let addr = spawn_test_server().await;
    let url = format!("http://{addr}/v1/chat/completions");

    let body = serde_json::json!({
        "model": "nonexistent",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await.expect("POST");
    assert_eq!(resp.status(), 404);

    let result: serde_json::Value = resp.json().await.expect("json");
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("nonexistent"));
}

#[tokio::test]
async fn test_empty_messages() {
    let addr = spawn_test_server().await;
    let url = format!("http://{addr}/v1/chat/completions");

    let body = serde_json::json!({
        "model": "test-model",
        "messages": []
    });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await.expect("POST");
    assert_eq!(resp.status(), 400);

    let result: serde_json::Value = resp.json().await.expect("json");
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("messages"));
}

#[tokio::test]
async fn test_streaming_completion() {
    let addr = spawn_test_server().await;
    let url = format!("http://{addr}/v1/chat/completions");

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "max_tokens": 3
    });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await.expect("POST");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.expect("response text");

    // Should contain SSE data lines
    assert!(text.contains("data: "), "Expected SSE data lines");

    // Should end with [DONE]
    assert!(text.contains("[DONE]"), "Expected [DONE] marker");

    // Parse each data line
    let mut saw_role = false;
    let mut saw_content = false;
    let mut saw_finish = false;

    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            let chunk: serde_json::Value = serde_json::from_str(data).expect("valid JSON chunk");
            let delta = &chunk["choices"][0]["delta"];
            if delta.get("role").is_some() {
                saw_role = true;
            }
            if delta.get("content").is_some() {
                saw_content = true;
            }
            if chunk["choices"][0]["finish_reason"].is_string() {
                saw_finish = true;
            }
        }
    }

    assert!(saw_role, "Expected role chunk");
    assert!(saw_content, "Expected content chunk");
    assert!(saw_finish, "Expected finish_reason chunk");
}
