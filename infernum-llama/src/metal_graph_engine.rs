//! Metal graph-mode inference engine for the Llama model family.
//!
//! Provides [`LlamaMetalGraphEngine`] (a type alias for
//! [`infernum_metal::MetalGraphEngine<LlamaConfig>`]) by implementing
//! [`infernum_metal::MetalGraphEngineConfig`] for [`LlamaConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightId, WeightStore};
use infernum::{DType, Result};
use infernum_metal::{
    load_graph_weights_metal, MetalBackend, MetalContext, MetalGraphEngineConfig, MetalLinearWeight,
    MetalTensor,
};

use crate::config::LlamaConfig;
use crate::graph_builder::{build_paged_decode_graph, build_prefill_graph, needs_unpermute, safetensors_to_gguf_name};

// ---------------------------------------------------------------------------
// MetalGraphEngineConfig impl
// ---------------------------------------------------------------------------

impl MetalGraphEngineConfig for LlamaConfig {
    infernum_metal::impl_metal_config_getters!();

    fn num_kv_heads(&self) -> usize {
        LlamaConfig::num_kv_heads(self)
    }

    fn head_dim(&self) -> usize {
        LlamaConfig::head_dim(self)
    }

    fn build_prefill_graph_metal(&self, seq_len: usize) -> Graph<infernum_metal::MetalBackend> {
        let (graph, _) =
            build_prefill_graph::<infernum_metal::MetalBackend>(self, seq_len, DType::F32);
        graph
    }

    fn build_paged_decode_graph_metal(
        &self,
        batch_size: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Graph<infernum_metal::MetalBackend> {
        build_paged_decode_graph::<infernum_metal::MetalBackend>(
            self,
            batch_size,
            block_size,
            max_blocks_per_seq,
            DType::F32,
        )
    }

    fn load_weights_metal_safetensors(
        &self,
        dummy_graph: &Graph<infernum_metal::MetalBackend>,
        ctx: &MetalContext,
        model_dir: &Path,
    ) -> Result<WeightStore<MetalTensor, MetalLinearWeight>> {
        // SmolLM2 and some other Llama-family models use tied embeddings:
        // `lm_head.weight` is absent and shares `model.embed_tokens.weight`.
        load_graph_weights_metal(
            dummy_graph,
            ctx,
            model_dir,
            /* lm_head_fallback */ true,
        )
    }
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// Load Llama model weights from a GGUF file into a Metal weight store.
///
/// Mirrors the CPU [`load_graph_weights_gguf`](crate::graph_builder::load_graph_weights_gguf)
/// but uploads tensors to the Metal device instead of the CPU backend.
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be opened, a required tensor is
/// missing, or an unsupported quantization type is encountered.
///
/// # Panics
///
/// Panics if the number of registered weights exceeds `u32::MAX`.
fn load_llama_graph_weights_gguf_metal(
    graph: &Graph<MetalBackend>,
    ctx: &MetalContext,
    gguf_path: &Path,
    num_attention_heads: usize,
    num_key_value_heads: usize,
) -> Result<WeightStore<MetalTensor, MetalLinearWeight>> {
    use infernum::backend::MatmulOps as _;
    use infernum::weights::format::{host_transpose_2d, host_unpermute_f32, FormatLoader};
    use infernum::weights::host::HostLinearWeight;

    let loader = infernum::weights::gguf::GgufLoader::from_file(infernum::path_to_utf8(gguf_path)?)?;

    let tensor_count = graph.tensor_weight_count();
    let linear_count = graph.linear_weight_count();
    let mut store = WeightStore::with_capacity(tensor_count, linear_count);

    for i in 0..tensor_count {
        let meta = graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let gguf_name = safetensors_to_gguf_name(&meta.name);
        let host = loader.load_f32(&gguf_name)?;
        store.push_tensor_weight(MetalTensor::from_f32(ctx, &host.shape, host.as_f32_slice()));
    }

    for i in 0..linear_count {
        let meta = graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let gguf_name = safetensors_to_gguf_name(&meta.name);

        let actual_name = if loader.contains(&gguf_name) {
            gguf_name.clone()
        } else if meta.name == "lm_head.weight" {
            "token_embd.weight".to_string()
        } else {
            return Err(infernum::Error::WeightNotFound(gguf_name));
        };

        let dtype = FormatLoader::get_dtype(&loader, &actual_name)?;
        let host_linear = if dtype.is_quantized() {
            if needs_unpermute(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    num_attention_heads
                } else {
                    num_key_value_heads
                };
                HostLinearWeight::Quantized(FormatLoader::load_quantized_unpermute(
                    &loader,
                    &actual_name,
                    n_head,
                )?)
            } else {
                HostLinearWeight::Quantized(FormatLoader::load_quantized(&loader, &actual_name)?)
            }
        } else {
            let host = loader.load_f32(&actual_name)?;
            let host = if needs_unpermute(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    num_attention_heads
                } else {
                    num_key_value_heads
                };
                host_unpermute_f32(&host, n_head)?
            } else {
                host
            };
            HostLinearWeight::Dense(host_transpose_2d(&host)?)
        };

        let linear = MetalBackend::upload_host_linear(ctx, &host_linear)?;
        store.push_linear_weight(linear);
    }

    Ok(store)
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// Metal graph-mode engine for Llama-family models.
///
/// A type alias for [`infernum_metal::MetalGraphEngine<LlamaConfig>`].
pub type LlamaMetalGraphEngine = infernum_metal::MetalGraphEngine<LlamaConfig>;

/// Extension trait providing Llama-specific Metal constructors.
pub trait LlamaMetalGraphEngineExt: Sized {
    /// Load a Llama-family model from a `SafeTensors` directory onto a Metal device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self>;

    /// Load a Llama-family model from a GGUF file onto a Metal device.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file cannot be opened, config metadata is
    /// missing, or a required weight cannot be loaded.
    fn from_gguf(ctx: MetalContext, gguf_path: &Path) -> Result<Self>;
}

impl LlamaMetalGraphEngineExt for LlamaMetalGraphEngine {
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self> {
        let config = LlamaConfig::from_file(model_dir.join("config.json"))?;
        infernum_metal::MetalGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }

    fn from_gguf(ctx: MetalContext, gguf_path: &Path) -> Result<Self> {
        let loader =
            infernum::weights::gguf::GgufLoader::from_file(infernum::path_to_utf8(gguf_path)?)?;
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;
        let dummy_graph = config.build_prefill_graph_metal(1);
        let weights = load_llama_graph_weights_gguf_metal(
            &dummy_graph,
            &ctx,
            gguf_path,
            config.num_attention_heads,
            config.num_kv_heads(),
        )?;
        Ok(infernum_metal::MetalGraphEngine::from_weights(config, ctx, weights))
    }
}
