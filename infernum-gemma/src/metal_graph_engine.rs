//! Metal graph-mode inference engine for the Gemma model family.
//!
//! Provides [`GemmaMetalGraphEngine`] (a type alias for
//! [`infernum_metal::MetalGraphEngine<GemmaConfig>`]) by implementing
//! [`infernum_metal::MetalGraphEngineConfig`] for [`GemmaConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightId, WeightStore};
use infernum::{DType, Result};
use infernum_metal::{
    load_graph_weights_metal, MetalBackend, MetalContext, MetalGraphEngineConfig,
    MetalLinearWeight, MetalTensor,
};

use crate::config::GemmaConfig;
use crate::graph_builder::{
    build_paged_decode_graph, build_prefill_graph, safetensors_to_gguf_name,
};

// ---------------------------------------------------------------------------
// MetalGraphEngineConfig impl
// ---------------------------------------------------------------------------

impl MetalGraphEngineConfig for GemmaConfig {
    infernum_metal::impl_metal_config_getters!();

    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn build_prefill_graph_metal(&self, seq_len: usize) -> Graph<infernum_metal::MetalBackend> {
        build_prefill_graph::<infernum_metal::MetalBackend>(self, seq_len, DType::F32, None)
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
            None,
        )
    }

    fn load_weights_metal_safetensors(
        &self,
        dummy_graph: &Graph<infernum_metal::MetalBackend>,
        ctx: &MetalContext,
        model_dir: &Path,
    ) -> Result<WeightStore<MetalTensor, MetalLinearWeight>> {
        // Gemma checkpoints use tied embeddings; lm_head.weight is absent and
        // the SafeTensors loader must fall back to model.embed_tokens.weight.
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

/// Load Gemma model weights from a GGUF file into a Metal weight store.
///
/// Unlike the Llama equivalent, Gemma weights do not require Q/K row
/// unpermutation — GGUF stores them in the same layout as HuggingFace.
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be opened, a required tensor is
/// missing, or an unsupported quantization type is encountered.
///
/// # Panics
///
/// Panics if the number of registered weights exceeds `u32::MAX`.
fn load_gemma_graph_weights_gguf_metal(
    graph: &Graph<MetalBackend>,
    ctx: &MetalContext,
    gguf_path: &Path,
) -> Result<WeightStore<MetalTensor, MetalLinearWeight>> {
    use infernum::backend::MatmulOps as _;
    use infernum::weights::format::{host_transpose_2d, FormatLoader};
    use infernum::weights::host::HostLinearWeight;

    let loader =
        infernum::weights::gguf::GgufLoader::from_file(infernum::path_to_utf8(gguf_path)?)?;

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
            HostLinearWeight::Quantized(FormatLoader::load_quantized(&loader, &actual_name)?)
        } else {
            let host = loader.load_f32(&actual_name)?;
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

/// Metal graph-mode engine for Gemma-family models.
///
/// A type alias for [`infernum_metal::MetalGraphEngine<GemmaConfig>`].
pub type GemmaMetalGraphEngine = infernum_metal::MetalGraphEngine<GemmaConfig>;

/// Extension trait providing Gemma-specific Metal constructors.
pub trait GemmaMetalGraphEngineExt: Sized {
    /// Load a Gemma-family model from a SafeTensors directory onto a Metal device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self>;

    /// Load a Gemma-family model from a GGUF file onto a Metal device.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file cannot be opened, config metadata is
    /// missing, or a required weight cannot be loaded.
    fn from_gguf(ctx: MetalContext, gguf_path: &Path) -> Result<Self>;
}

impl GemmaMetalGraphEngineExt for GemmaMetalGraphEngine {
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self> {
        let config = GemmaConfig::from_file(model_dir.join("config.json"))?;
        infernum_metal::MetalGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }

    fn from_gguf(ctx: MetalContext, gguf_path: &Path) -> Result<Self> {
        let loader =
            infernum::weights::gguf::GgufLoader::from_file(infernum::path_to_utf8(gguf_path)?)?;
        let config = GemmaConfig::from_gguf_metadata(loader.metadata())?;
        let dummy_graph = config.build_prefill_graph_metal(1);
        let weights = load_gemma_graph_weights_gguf_metal(&dummy_graph, &ctx, gguf_path)?;
        Ok(infernum_metal::MetalGraphEngine::from_weights(
            config, ctx, weights,
        ))
    }
}
