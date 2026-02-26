//! Backend-generic HTTP server implementation.
//!
//! Provides the [`Server`] builder for registering models and running
//! the OpenAI-compatible API. No CUDA-specific imports.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use infernum::chat_template::ChatTemplate;
use infernum::{
    ChatMessage, GenerateOptions, Model, ModelConfig, Result as InfernumResult, SamplingParams,
    Tokenizer,
};
use infernum_runtime::{BatchConfig, Engine, FinishReason, GenerationEvent, TokenSender};

use crate::types::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, ChatMessage as ApiChatMessage, ErrorBody, ErrorResponse,
    ModelListResponse, ModelObject, Usage,
};

// ---------------------------------------------------------------------------
// TokenSender impl for tokio channels (newtype to satisfy orphan rules)
// ---------------------------------------------------------------------------

/// Wrapper around a tokio MPSC sender that implements [`TokenSender`].
///
/// Uses `blocking_send`, which is safe because the engine's worker thread
/// is a plain `std::thread` (not a tokio task).
struct TokioTokenSender(tokio_mpsc::Sender<GenerationEvent>);

impl TokenSender for TokioTokenSender {
    fn send(&self, event: GenerationEvent) -> bool {
        self.0.blocking_send(event).is_ok()
    }
}

// ---------------------------------------------------------------------------
// ModelHandle (type-erased engine + tokenizer + template)
// ---------------------------------------------------------------------------

/// Object-safe subset of [`Tokenizer`] needed by the server.
#[allow(dead_code)]
trait ErasedTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool) -> InfernumResult<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> InfernumResult<String>;
    fn decode_token(&self, id: u32) -> InfernumResult<String>;
    fn eos_token_id(&self) -> u32;
}

impl<T: Tokenizer + Send + Sync> ErasedTokenizer for T {
    fn encode(&self, text: &str, add_bos: bool) -> InfernumResult<Vec<u32>> {
        Tokenizer::encode(self, text, add_bos)
    }
    fn decode(&self, ids: &[u32]) -> InfernumResult<String> {
        Tokenizer::decode(self, ids)
    }
    fn decode_token(&self, id: u32) -> InfernumResult<String> {
        Tokenizer::decode_token(self, id)
    }
    fn eos_token_id(&self) -> u32 {
        Tokenizer::eos_token_id(self)
    }
}

/// Object-safe engine interface for submitting generation requests.
#[allow(dead_code)]
trait ErasedEngine: Send + Sync {
    fn submit(&self, input_ids: Vec<u32>, options: GenerateOptions, token_tx: Box<dyn TokenSender>);
    fn model_config(&self) -> &ModelConfig;
}

impl<M: Model> ErasedEngine for Engine<M> {
    fn submit(
        &self,
        input_ids: Vec<u32>,
        options: GenerateOptions,
        token_tx: Box<dyn TokenSender>,
    ) {
        Engine::submit(self, input_ids, options, token_tx);
    }

    fn model_config(&self) -> &ModelConfig {
        Engine::model_config(self)
    }
}

/// Handle to a model, its engine, tokenizer, and chat template.
struct ModelHandle {
    engine: Box<dyn ErasedEngine>,
    tokenizer: Box<dyn ErasedTokenizer>,
    template: Box<dyn ChatTemplate>,
    model_config: ModelConfig,
}

// ---------------------------------------------------------------------------
// ModelEntry — user-facing model registration
// ---------------------------------------------------------------------------

/// Entry point for registering a model with the server.
///
/// Bundles a model, tokenizer, and chat template under a name. The model
/// is type-erased so the server doesn't need concrete model types.
pub struct ModelEntry {
    name: String,
    handle: ModelHandle,
}

impl ModelEntry {
    /// Create a new model entry with default batch configuration.
    ///
    /// The model is consumed and moved into a background engine thread.
    ///
    /// # Panics
    /// Panics if engine creation fails (paged KV cache allocation).
    pub fn new<M, T, C>(name: &str, model: M, tokenizer: T, template: C) -> Self
    where
        M: Model,
        T: Tokenizer + Send + Sync + 'static,
        C: ChatTemplate + 'static,
    {
        let model_config = model.config();
        let engine = Engine::new(model).expect("Failed to create engine");
        Self {
            name: name.to_string(),
            handle: ModelHandle {
                engine: Box::new(engine),
                tokenizer: Box::new(tokenizer),
                template: Box::new(template),
                model_config,
            },
        }
    }

    /// Create a new model entry with custom batch configuration.
    ///
    /// # Panics
    /// Panics if engine creation fails (paged KV cache allocation).
    pub fn with_config<M, T, C>(
        name: &str,
        model: M,
        tokenizer: T,
        template: C,
        batch_config: BatchConfig,
    ) -> Self
    where
        M: Model,
        T: Tokenizer + Send + Sync + 'static,
        C: ChatTemplate + 'static,
    {
        let model_config = model.config();
        let engine = Engine::with_config(model, batch_config).expect("Failed to create engine");
        Self {
            name: name.to_string(),
            handle: ModelHandle {
                engine: Box::new(engine),
                tokenizer: Box::new(tokenizer),
                template: Box::new(template),
                model_config,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// AppState — shared server state
// ---------------------------------------------------------------------------

struct AppState2 {
    models: HashMap<String, Arc<ModelHandle>>,
}

// ---------------------------------------------------------------------------
// Server + Builder
// ---------------------------------------------------------------------------

/// Backend-generic HTTP server.
pub struct Server {
    bind_addr: SocketAddr,
    state: Arc<AppState2>,
}

/// Builder for constructing a [`Server`].
pub struct ServerBuilder {
    models: HashMap<String, Arc<ModelHandle>>,
    bind_addr: SocketAddr,
}

impl Server {
    /// Create a new server builder.
    #[must_use]
    pub fn builder() -> ServerBuilder {
        ServerBuilder {
            models: HashMap::new(),
            bind_addr: SocketAddr::from(([0, 0, 0, 0], 8080)),
        }
    }

    /// Run the server, binding to the configured address.
    ///
    /// # Errors
    /// Returns an error if the server fails to bind or run.
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind(self.bind_addr).await?;
        eprintln!("Listening on {}", self.bind_addr);
        self.run_on(listener).await
    }

    /// Run the server on a pre-bound listener.
    ///
    /// Useful for tests that need to bind to a random port.
    ///
    /// # Errors
    /// Returns an error if the server fails to run.
    pub async fn run_on(
        self,
        listener: tokio::net::TcpListener,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let app = build_router2(self.state);
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        Ok(())
    }
}

impl ServerBuilder {
    /// Register a model with the server.
    #[must_use]
    pub fn add_model(mut self, entry: ModelEntry) -> Self {
        self.models.insert(entry.name, Arc::new(entry.handle));
        self
    }

    /// Set the address to bind the server to.
    #[must_use]
    pub fn bind(mut self, addr: impl Into<SocketAddr>) -> Self {
        self.bind_addr = addr.into();
        self
    }

    /// Build the server.
    #[must_use]
    pub fn build(self) -> Server {
        Server {
            bind_addr: self.bind_addr,
            state: Arc::new(AppState2 {
                models: self.models,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

fn build_router2(state: Arc<AppState2>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler2))
        .route("/v1/models", get(list_models_handler2))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

async fn list_models_handler2(State(state): State<Arc<AppState2>>) -> Json<ModelListResponse> {
    let now = unix_timestamp();
    let data = state
        .models
        .keys()
        .map(|name| ModelObject {
            id: name.clone(),
            object: "model".into(),
            created: now,
            owned_by: "infernum".into(),
        })
        .collect();
    Json(ModelListResponse {
        object: "list".into(),
        data,
    })
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

async fn chat_completions_handler2(
    State(state): State<Arc<AppState2>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Look up model
    let handle = match state.models.get(&request.model) {
        Some(h) => Arc::clone(h),
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model '{}' not found", request.model),
                "invalid_request_error",
                Some("model"),
                Some("model_not_found"),
            );
        }
    };

    // Validate messages
    if request.messages.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            Some("messages"),
            None,
        );
    }

    // Apply chat template
    let chat_messages: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let prompt = handle.template.apply(&chat_messages);

    // Tokenize
    let input_ids = match handle.tokenizer.encode(&prompt, false) {
        Ok(ids) => ids,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Tokenization failed: {e}"),
                "server_error",
                None,
                None,
            );
        }
    };

    let prompt_tokens = input_ids.len();

    // Build generation options
    let options = build_generate_options2(&request, &handle);

    let stream = request.stream.unwrap_or(false);

    if stream {
        handle_streaming2(handle, input_ids, options, &request.model).into_response()
    } else {
        handle_non_streaming2(handle, input_ids, options, prompt_tokens, &request.model)
            .await
            .into_response()
    }
}

// ---------------------------------------------------------------------------
// Non-streaming response
// ---------------------------------------------------------------------------

async fn handle_non_streaming2(
    handle: Arc<ModelHandle>,
    input_ids: Vec<u32>,
    options: GenerateOptions,
    prompt_tokens: usize,
    model_name: &str,
) -> Response {
    let (tx, mut rx) = tokio_mpsc::channel::<GenerationEvent>(256);
    handle
        .engine
        .submit(input_ids, options, Box::new(TokioTokenSender(tx)));

    let mut generated_ids: Vec<u32> = Vec::new();
    let mut finish_reason = "length";

    while let Some(event) = rx.recv().await {
        match event {
            GenerationEvent::Token(id) => generated_ids.push(id),
            GenerationEvent::Error(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Generation error: {e}"),
                    "server_error",
                    None,
                    None,
                );
            }
            GenerationEvent::Finished(reason) => {
                finish_reason = match reason {
                    FinishReason::Stop => "stop",
                    FinishReason::Length | FinishReason::Cancelled => "length",
                };
                break;
            }
        }
    }

    // Decode generated tokens
    let content = match handle.tokenizer.decode(&generated_ids) {
        Ok(text) => text,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Decoding error: {e}"),
                "server_error",
                None,
                None,
            );
        }
    };

    let completion_tokens = generated_ids.len();

    let response = ChatCompletionResponse {
        id: generate_id(),
        object: "chat.completion".into(),
        created: unix_timestamp(),
        model: model_name.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ApiChatMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason: finish_reason.into(),
        }],
        usage: Usage {
            prompt_tokens: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(completion_tokens).unwrap_or(u32::MAX),
            total_tokens: u32::try_from(prompt_tokens + completion_tokens).unwrap_or(u32::MAX),
        },
    };

    Json(response).into_response()
}

// ---------------------------------------------------------------------------
// Streaming (SSE) response
// ---------------------------------------------------------------------------

fn handle_streaming2(
    handle: Arc<ModelHandle>,
    input_ids: Vec<u32>,
    options: GenerateOptions,
    model_name: &str,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (engine_tx, mut engine_rx) = tokio_mpsc::channel::<GenerationEvent>(256);
    handle
        .engine
        .submit(input_ids, options, Box::new(TokioTokenSender(engine_tx)));

    let (sse_tx, sse_rx) = tokio_mpsc::channel::<Result<Event, std::convert::Infallible>>(256);

    let id = generate_id();
    let created = unix_timestamp();
    let model = model_name.to_string();

    tokio::spawn(async move {
        // First chunk: role
        let first = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".into()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        if send_sse_chunk(&sse_tx, &first).await.is_err() {
            return;
        }

        let mut finish = "length";
        while let Some(event) = engine_rx.recv().await {
            match event {
                GenerationEvent::Token(token) => {
                    let Ok(text) = handle.tokenizer.decode_token(token) else {
                        continue;
                    };

                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    if send_sse_chunk(&sse_tx, &chunk).await.is_err() {
                        return;
                    }
                }
                GenerationEvent::Error(_) => break,
                GenerationEvent::Finished(reason) => {
                    finish = match reason {
                        FinishReason::Stop => "stop",
                        FinishReason::Length | FinishReason::Cancelled => "length",
                    };
                    break;
                }
            }
        }
        let final_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish.into()),
            }],
        };
        let _ = send_sse_chunk(&sse_tx, &final_chunk).await;

        // data: [DONE]
        let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = ReceiverStream::new(sse_rx);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn send_sse_chunk(
    tx: &tokio_mpsc::Sender<Result<Event, std::convert::Infallible>>,
    chunk: &ChatCompletionChunk,
) -> Result<(), ()> {
    let json = serde_json::to_string(chunk).map_err(|_| ())?;
    tx.send(Ok(Event::default().data(json)))
        .await
        .map_err(|_| ())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_generate_options2(
    request: &ChatCompletionRequest,
    handle: &ModelHandle,
) -> GenerateOptions {
    let max_new_tokens = request.max_tokens.unwrap_or(256);

    let sampling = if request.temperature.is_some() || request.top_p.is_some() {
        Some(SamplingParams {
            temperature: request.temperature.unwrap_or(0.7),
            top_p: request.top_p.unwrap_or(0.9),
            seed: request.seed.unwrap_or(42),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            repetition_penalty_window: 64,
        })
    } else {
        None
    };

    GenerateOptions {
        max_new_tokens,
        eos_token_id: Some(handle.model_config.eos_token_id),
        sampling,
        use_kv_cache: true,
    }
}

fn error_response(
    status: StatusCode,
    message: &str,
    error_type: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    let body = ErrorResponse {
        error: ErrorBody {
            message: message.into(),
            error_type: error_type.into(),
            param: param.map(String::from),
            code: code.map(String::from),
        },
    };
    (status, Json(body)).into_response()
}

fn generate_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    eprintln!(
        "
Shutting down..."
    );
}
