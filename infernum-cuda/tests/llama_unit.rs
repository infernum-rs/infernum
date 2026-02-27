//! CUDA unit tests for LlamaModel: forward pass, generation, GPTQ quantization,
//! and linear op correctness. These use synthetic tiny models (no downloads).
//!
//! Gated behind `cuda` feature — requires a CUDA GPU.

#![cfg(feature = "cuda")]
#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;

use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;
use infernum_cuda::cuda::ops::{linear, LinearWeight};
use infernum_cuda::cuda::{CudaContext, CudaTensor, QuantizedTensor};
use infernum_cuda::weights::{CudaWeightLoader, WeightLoader};
use infernum_cuda::CudaBackend;
use infernum_llama::{LlamaConfig, LlamaModel, QuantizationConfig};

// ---- Test helpers ----

/// Deterministic pseudo-random f32 in [-scale, scale] for reproducible test weights
fn pseudo_random_weights(n: usize, scale: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(n);
    let mut state: u64 = 42;
    for _ in 0..n {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let f = (state as f32) / (u64::MAX as f32); // [0, 1)
        values.push((f * 2.0 - 1.0) * scale);
    }
    values
}

/// Stored GPTQ data for a single linear layer
struct GptqLayerData {
    shape: Vec<usize>,
    qweight: Vec<u8>,
    scales: Vec<u8>,
    qzeros: Vec<u8>,
    group_size: usize,
}

struct MockWeightLoader {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    gptq_weights: HashMap<String, GptqLayerData>,
}

impl MockWeightLoader {
    fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            gptq_weights: HashMap::new(),
        }
    }

    fn add(&mut self, name: &str, shape: &[usize], data: Vec<f32>) {
        self.tensors
            .insert(name.to_string(), (shape.to_vec(), data));
    }

    fn add_gptq(
        &mut self,
        prefix: &str,
        shape: &[usize],
        qweight: Vec<u8>,
        scales: Vec<u8>,
        qzeros: Vec<u8>,
        group_size: usize,
    ) {
        self.gptq_weights.insert(
            prefix.to_string(),
            GptqLayerData {
                shape: shape.to_vec(),
                qweight,
                scales,
                qzeros,
                group_size,
            },
        );
    }
}

impl WeightLoader for MockWeightLoader {
    fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
        let (shape, data) = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("MockWeightLoader: tensor not found: {name}"));
        CudaTensor::from_slice(ctx, shape, data)
    }

    fn load_f16(&self, _ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
        Err(infernum::Error::UnsupportedDtype(format!(
            "MockWeightLoader: load_f16 not supported (tensor: {name})"
        )))
    }

    fn load_bf16(&self, _ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
        Err(infernum::Error::UnsupportedDtype(format!(
            "MockWeightLoader: load_bf16 not supported (tensor: {name})"
        )))
    }

    fn load_gptq_linear(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        _group_size: usize,
    ) -> Result<QuantizedTensor> {
        let data = self
            .gptq_weights
            .get(prefix)
            .unwrap_or_else(|| panic!("MockWeightLoader: GPTQ prefix not found: {prefix}"));
        QuantizedTensor::from_gptq_raw(
            ctx,
            &data.shape,
            DType::GPTQ_INT4,
            &data.qweight,
            &data.scales,
            &data.qzeros,
            data.group_size,
        )
    }

    fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensors.get(name).unwrap().0.clone())
    }

    fn get_dtype(&self, _name: &str) -> Result<DType> {
        Ok(DType::F32)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

fn tiny_config() -> LlamaConfig {
    LlamaConfig {
        model_type: "llama".to_string(),
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: Some(2),
        max_position_embeddings: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        bos_token_id: 1,
        eos_token_id: 2,
        quantization_config: None,
        num_local_experts: None,
        num_experts_per_tok: None,
        sliding_window: None,
        use_sliding_window: false,
        max_window_layers: None,
    }
}

fn tiny_weight_loader(config: &LlamaConfig) -> MockWeightLoader {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let scale = 0.02;

    let mut loader = MockWeightLoader::new();
    let mut seed_offset: usize = 0;

    let mut rand = |n: usize| -> Vec<f32> {
        seed_offset += 1;
        let mut vals = pseudo_random_weights(n, scale);
        vals.rotate_left(seed_offset % n.max(1));
        vals
    };

    loader.add("model.embed_tokens.weight", &[vocab, h], rand(vocab * h));

    let prefix = "model.layers.0";
    loader.add(
        &format!("{prefix}.input_layernorm.weight"),
        &[h],
        vec![1.0; h],
    );
    loader.add(
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[num_heads * head_dim, h],
        rand(num_heads * head_dim * h),
    );
    loader.add(
        &format!("{prefix}.self_attn.k_proj.weight"),
        &[num_kv_heads * head_dim, h],
        rand(num_kv_heads * head_dim * h),
    );
    loader.add(
        &format!("{prefix}.self_attn.v_proj.weight"),
        &[num_kv_heads * head_dim, h],
        rand(num_kv_heads * head_dim * h),
    );
    loader.add(
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[h, num_heads * head_dim],
        rand(h * num_heads * head_dim),
    );
    loader.add(
        &format!("{prefix}.post_attention_layernorm.weight"),
        &[h],
        vec![1.0; h],
    );
    loader.add(
        &format!("{prefix}.mlp.gate_proj.weight"),
        &[inter, h],
        rand(inter * h),
    );
    loader.add(
        &format!("{prefix}.mlp.up_proj.weight"),
        &[inter, h],
        rand(inter * h),
    );
    loader.add(
        &format!("{prefix}.mlp.down_proj.weight"),
        &[h, inter],
        rand(h * inter),
    );

    loader.add("model.norm.weight", &[h], vec![1.0; h]);
    loader.add("lm_head.weight", &[vocab, h], rand(vocab * h));

    loader
}

fn build_tiny_model(ctx: &CudaContext) -> LlamaModel<CudaBackend> {
    let config = tiny_config();
    let format_loader = tiny_weight_loader(&config);
    let loader = CudaWeightLoader::new(ctx.clone(), format_loader);
    LlamaModel::load_weights(ctx.clone(), config, &loader).expect("Failed to build tiny model")
}

/// Pack f32 weights into GPTQ INT4 format on the host.
fn pack_gptq_test(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert_eq!(weights.len(), out_features * in_features);
    assert_eq!(in_features % 8, 0);
    assert_eq!(in_features % group_size, 0);
    assert_eq!(out_features % 8, 0);

    let num_groups = in_features / group_size;
    let packed_rows = in_features / 8;
    let zero_point = 8_i32;

    let mut scales_f16 = vec![half::f16::from_f32(0.0); num_groups * out_features];
    let mut quantized = vec![0_i32; out_features * in_features];

    for n in 0..out_features {
        for g in 0..num_groups {
            let k_start = g * group_size;
            let k_end = k_start + group_size;
            let group_vals: Vec<f32> = (k_start..k_end)
                .map(|k| weights[n * in_features + k])
                .collect();
            let max_abs = group_vals.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            scales_f16[g * out_features + n] = half::f16::from_f32(scale);

            for (j, &v) in group_vals.iter().enumerate() {
                let q = ((v / scale).round() as i32 + zero_point).clamp(0, 15);
                quantized[n * in_features + k_start + j] = q;
            }
        }
    }

    // Pack qweight: [in_features/8, out_features] as int32
    let mut qweight = vec![0_u8; packed_rows * out_features * 4];
    for pr in 0..packed_rows {
        for n in 0..out_features {
            let mut packed: u32 = 0;
            for j in 0..8 {
                let k = pr * 8 + j;
                let q = quantized[n * in_features + k] as u32;
                packed |= (q & 0xF) << (j * 4);
            }
            let idx = (pr * out_features + n) * 4;
            qweight[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
        }
    }

    // Pack scales: [num_groups, out_features] as f16
    let mut scales_bytes = vec![0_u8; num_groups * out_features * 2];
    for (i, &s) in scales_f16.iter().enumerate() {
        let bytes = s.to_le_bytes();
        scales_bytes[i * 2] = bytes[0];
        scales_bytes[i * 2 + 1] = bytes[1];
    }

    // Pack qzeros: [num_groups, out_features/8] as int32
    let qzeros_cols = out_features / 8;
    let mut qzeros = vec![0_u8; num_groups * qzeros_cols * 4];
    for g in 0..num_groups {
        for col in 0..qzeros_cols {
            let mut packed: u32 = 0;
            for j in 0..8 {
                packed |= (zero_point as u32 & 0xF) << (j * 4);
            }
            let idx = (g * qzeros_cols + col) * 4;
            qzeros[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
        }
    }

    (qweight, scales_bytes, qzeros)
}

fn tiny_gptq_config() -> LlamaConfig {
    LlamaConfig {
        quantization_config: Some(QuantizationConfig {
            quant_method: "gptq".to_string(),
            bits: 4,
            group_size: 32,
        }),
        ..tiny_config()
    }
}

fn rand_gptq(
    seed_offset: &mut usize,
    out_f: usize,
    in_f: usize,
    group_size: usize,
    scale: f32,
) -> (Vec<usize>, Vec<u8>, Vec<u8>, Vec<u8>) {
    *seed_offset += 1;
    let mut w = pseudo_random_weights(out_f * in_f, scale);
    w.rotate_left(*seed_offset % (out_f * in_f).max(1));
    let (qw, sc, qz) = pack_gptq_test(&w, out_f, in_f, group_size);
    (vec![out_f, in_f], qw, sc, qz)
}

fn rand_f32(seed_offset: &mut usize, n: usize, scale: f32) -> Vec<f32> {
    *seed_offset += 1;
    let mut vals = pseudo_random_weights(n, scale);
    vals.rotate_left(*seed_offset % n.max(1));
    vals
}

fn tiny_gptq_weight_loader(config: &LlamaConfig) -> MockWeightLoader {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let group_size = config.quantization_config.as_ref().unwrap().group_size;
    let scale = 0.02;

    let mut loader = MockWeightLoader::new();
    let mut seed_offset: usize = 0;

    loader.add(
        "model.embed_tokens.weight",
        &[vocab, h],
        rand_f32(&mut seed_offset, vocab * h, scale),
    );

    let prefix = "model.layers.0";
    loader.add(
        &format!("{prefix}.input_layernorm.weight"),
        &[h],
        vec![1.0; h],
    );

    let gptq_layers: Vec<(&str, usize, usize)> = vec![
        ("self_attn.q_proj", num_heads * head_dim, h),
        ("self_attn.k_proj", num_kv_heads * head_dim, h),
        ("self_attn.v_proj", num_kv_heads * head_dim, h),
        ("self_attn.o_proj", h, num_heads * head_dim),
        ("mlp.gate_proj", inter, h),
        ("mlp.up_proj", inter, h),
        ("mlp.down_proj", h, inter),
    ];

    for (name, out_f, in_f) in &gptq_layers {
        let (shape, qw, sc, qz) = rand_gptq(&mut seed_offset, *out_f, *in_f, group_size, scale);
        loader.add_gptq(&format!("{prefix}.{name}"), &shape, qw, sc, qz, group_size);
    }

    loader.add(
        &format!("{prefix}.post_attention_layernorm.weight"),
        &[h],
        vec![1.0; h],
    );

    loader.add("model.norm.weight", &[h], vec![1.0; h]);
    loader.add(
        "lm_head.weight",
        &[vocab, h],
        rand_f32(&mut seed_offset, vocab * h, scale),
    );

    loader
}

fn build_tiny_gptq_model(ctx: &CudaContext) -> LlamaModel<CudaBackend> {
    let config = tiny_gptq_config();
    let format_loader = tiny_gptq_weight_loader(&config);
    let loader = CudaWeightLoader::new(ctx.clone(), format_loader);
    LlamaModel::load_weights(ctx.clone(), config, &loader).expect("Failed to build tiny GPTQ model")
}

fn build_tiny_engine(ctx: &CudaContext) -> infernum_runtime::Engine<LlamaModel<CudaBackend>> {
    let model = build_tiny_model(ctx);
    infernum_runtime::Engine::new(model).expect("Failed to build engine")
}

// ---- Linear op tests ----

#[test]
fn test_linear() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 1.0, // row 0
        0.0, 1.0, 0.0, 1.0, // row 1
        0.0, 0.0, 1.0, 1.0, // row 2
    ];

    let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
    let weight = LinearWeight::Dense(CudaTensor::from_slice(&ctx, &[3, 4], &weight_data).unwrap());

    let output = linear(&input, &weight).unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    let result = output.to_vec::<f32>().unwrap();
    assert!((result[0] - 1.0).abs() < 1e-4);
    assert!((result[1] - 2.0).abs() < 1e-4);
    assert!((result[2] - 3.0).abs() < 1e-4);
    assert!((result[3] - 6.0).abs() < 1e-4);
    assert!((result[4] - 4.0).abs() < 1e-4);
    assert!((result[5] - 5.0).abs() < 1e-4);
    assert!((result[6] - 6.0).abs() < 1e-4);
    assert!((result[7] - 15.0).abs() < 1e-4);
}

#[test]
fn test_linear_bf16() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    let input_data: Vec<half::bf16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let weight_data: Vec<half::bf16> = vec![
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(1.0),
    ];

    let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
    let weight = LinearWeight::Dense(CudaTensor::from_slice(&ctx, &[3, 4], &weight_data).unwrap());

    let output = linear(&input, &weight).unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    let result: Vec<f32> = output
        .to_vec::<half::bf16>()
        .unwrap()
        .into_iter()
        .map(half::bf16::to_f32)
        .collect();
    assert!((result[0] - 1.0).abs() < 0.1);
    assert!((result[1] - 2.0).abs() < 0.1);
    assert!((result[2] - 3.0).abs() < 0.1);
    assert!((result[3] - 6.0).abs() < 0.1);
    assert!((result[4] - 4.0).abs() < 0.1);
    assert!((result[5] - 5.0).abs() < 0.1);
    assert!((result[6] - 6.0).abs() < 0.1);
    assert!((result[7] - 15.0).abs() < 0.2);
}

#[test]
fn test_linear_gptq() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    let k = 32;
    let n = 8;
    let group_size = 32;

    let w_data = vec![1.0_f32; n * k];
    let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

    let weight = LinearWeight::Quantized(
        QuantizedTensor::from_gptq_raw(&ctx, &[n, k], DType::GPTQ_INT4, &qw, &sc, &qz, group_size)
            .unwrap(),
    );

    let input_data = vec![1.0_f32; k];
    let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();

    let output = linear(&input, &weight).unwrap();
    assert_eq!(output.shape(), &[1, n]);

    let result = output.to_vec::<f32>().unwrap();
    let expected = 32.0_f32;
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < expected * 0.15,
            "GPTQ linear [{i}]: {v} vs expected ~{expected}",
        );
    }
}

// ---- Forward pass tests (dense) ----

#[test]
fn test_forward_output_shape() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = build_tiny_model(&ctx);

    let input_ids: Vec<u32> = vec![1, 5, 10];
    let logits = model.forward_full(&input_ids).expect("Forward pass failed");
    assert_eq!(logits.shape(), &[3, 64]);
}

#[test]
fn test_forward_single_token() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = build_tiny_model(&ctx);

    let logits = model.forward_full(&[0]).expect("Forward pass failed");
    assert_eq!(logits.shape(), &[1, 64]);

    let data = logits.to_vec::<f32>().unwrap();
    assert!(
        data.iter().all(|x| x.is_finite()),
        "Logits contain non-finite values"
    );
}

#[test]
fn test_forward_deterministic() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = build_tiny_model(&ctx);

    let input_ids: Vec<u32> = vec![1, 5, 10];
    let logits1 = model
        .forward_full(&input_ids)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let logits2 = model
        .forward_full(&input_ids)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();

    for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "Non-deterministic at index {i}: {a} vs {b}"
        );
    }
}

// ---- Forward pass tests (GPTQ) ----

#[test]
fn test_forward_gptq_output_shape() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = build_tiny_gptq_model(&ctx);

    let input_ids: Vec<u32> = vec![1, 5, 10];
    let logits = model.forward_full(&input_ids).expect("Forward pass failed");
    assert_eq!(logits.shape(), &[3, 64]);
}

#[test]
fn test_forward_gptq_single_token() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = build_tiny_gptq_model(&ctx);

    let logits = model.forward_full(&[0]).expect("Forward pass failed");
    assert_eq!(logits.shape(), &[1, 64]);

    let data = logits.to_vec::<f32>().unwrap();
    assert!(
        data.iter().all(|x| x.is_finite()),
        "Logits contain non-finite values"
    );
}

#[test]
fn test_forward_gptq_deterministic() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = build_tiny_gptq_model(&ctx);

    let input_ids: Vec<u32> = vec![1, 5, 10];
    let logits1 = model
        .forward_full(&input_ids)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let logits2 = model
        .forward_full(&input_ids)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();

    for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "GPTQ non-deterministic at index {i}: {a} vs {b}"
        );
    }
}

// ---- Generation tests ----

#[test]
fn test_generate_respects_max_tokens() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let engine = build_tiny_engine(&ctx);

    let prompt = vec![1_u32, 5, 10];
    let max_new = 4;
    let options = infernum::GenerateOptions {
        max_new_tokens: max_new,
        use_kv_cache: false,
        ..Default::default()
    };
    let tokens = engine.generate(&prompt, &options).unwrap();

    assert!(tokens.len() <= prompt.len() + max_new);
    assert!(
        tokens.len() > prompt.len(),
        "Should generate at least 1 token"
    );
    assert_eq!(&tokens[..prompt.len()], &prompt);
}

#[test]
fn test_generate_gptq_respects_max_tokens() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = build_tiny_gptq_model(&ctx);
    let engine = infernum_runtime::Engine::new(model).expect("Failed to build engine");

    let prompt = vec![1_u32, 5, 10];
    let max_new = 4;
    let options = infernum::GenerateOptions {
        max_new_tokens: max_new,
        use_kv_cache: false,
        ..Default::default()
    };
    let tokens = engine.generate(&prompt, &options).unwrap();

    assert!(tokens.len() <= prompt.len() + max_new);
    assert!(
        tokens.len() > prompt.len(),
        "Should generate at least 1 token"
    );
    assert_eq!(&tokens[..prompt.len()], &prompt);
}

#[test]
fn test_generate_stops_on_eos() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let engine = build_tiny_engine(&ctx);

    let prompt = vec![1_u32];
    let options_no_eos = infernum::GenerateOptions {
        max_new_tokens: 5,
        use_kv_cache: false,
        ..Default::default()
    };
    let result_no_eos = engine.generate(&prompt, &options_no_eos).unwrap();

    if result_no_eos.len() > 1 {
        let first_generated = result_no_eos[1];
        let options_with_eos = infernum::GenerateOptions {
            max_new_tokens: 100,
            eos_token_id: Some(first_generated),
            use_kv_cache: false,
            ..Default::default()
        };
        let result_with_eos = engine.generate(&prompt, &options_with_eos).unwrap();
        assert_eq!(
            result_with_eos.len(),
            prompt.len(),
            "Should stop before appending the EOS token"
        );
    }
}

#[test]
fn test_kv_cache_generate_matches_naive_generate() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let engine = build_tiny_engine(&ctx);

    let prompt: Vec<u32> = vec![1, 5, 10];
    let max_new = 5;

    let naive_options = infernum::GenerateOptions {
        max_new_tokens: max_new,
        use_kv_cache: false,
        ..Default::default()
    };
    let tokens_naive = engine.generate(&prompt, &naive_options).unwrap();

    let kv_options = infernum::GenerateOptions {
        max_new_tokens: max_new,
        use_kv_cache: true,
        ..Default::default()
    };
    let tokens_kv = engine.generate(&prompt, &kv_options).unwrap();

    assert_eq!(
        tokens_naive, tokens_kv,
        "KV-cache generate should match naive generate"
    );
}

// ---- GPTQ sharding tests ----

/// Slice columns from a 2D row-major byte buffer.
fn slice_columns(
    data: &[u8],
    rows: usize,
    total_cols: usize,
    col_start: usize,
    col_count: usize,
    elem_bytes: usize,
) -> Vec<u8> {
    let row_bytes = total_cols * elem_bytes;
    let col_start_bytes = col_start * elem_bytes;
    let col_count_bytes = col_count * elem_bytes;
    let mut result = Vec::with_capacity(rows * col_count_bytes);
    for r in 0..rows {
        let off = r * row_bytes + col_start_bytes;
        result.extend_from_slice(&data[off..off + col_count_bytes]);
    }
    result
}

/// Slice contiguous rows from a 2D row-major byte buffer.
fn slice_rows(
    data: &[u8],
    cols: usize,
    row_start: usize,
    row_count: usize,
    elem_bytes: usize,
) -> Vec<u8> {
    let row_bytes = cols * elem_bytes;
    let start = row_start * row_bytes;
    data[start..start + row_count * row_bytes].to_vec()
}

#[test]
fn test_linear_gptq_column_shard() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    let k = 32;
    let n = 16;
    let group_size = 32;
    let world_size = 2;

    let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
    let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

    let full_weight = LinearWeight::Quantized(
        QuantizedTensor::from_gptq_raw(&ctx, &[n, k], DType::GPTQ_INT4, &qw, &sc, &qz, group_size)
            .unwrap(),
    );
    let input_data: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
    let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();
    let full_output = linear(&input, &full_weight)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();

    let mut shard_outputs = Vec::new();
    for rank in 0..world_size {
        let n_shard = n / world_size;
        let n_start = rank * n_shard;

        let qw_s = slice_columns(&qw, k / 8, n, n_start, n_shard, 4);
        let sc_s = slice_columns(&sc, k / group_size, n, n_start, n_shard, 2);
        let qz_s = slice_columns(&qz, k / group_size, n / 8, n_start / 8, n_shard / 8, 4);

        let shard_weight = LinearWeight::Quantized(
            QuantizedTensor::from_gptq_raw(
                &ctx,
                &[n_shard, k],
                DType::GPTQ_INT4,
                &qw_s,
                &sc_s,
                &qz_s,
                group_size,
            )
            .unwrap(),
        );
        let shard_out = linear(&input, &shard_weight)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(shard_out.len(), n_shard);
        shard_outputs.extend(shard_out);
    }

    assert_eq!(shard_outputs.len(), full_output.len());
    for (i, (a, b)) in full_output.iter().zip(shard_outputs.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "Column shard mismatch at {i}: full={a} vs sharded={b}"
        );
    }
}

#[test]
fn test_linear_gptq_row_shard() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    let k = 64;
    let n = 8;
    let group_size = 32;
    let world_size = 2;

    let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
    let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

    let full_weight = LinearWeight::Quantized(
        QuantizedTensor::from_gptq_raw(&ctx, &[n, k], DType::GPTQ_INT4, &qw, &sc, &qz, group_size)
            .unwrap(),
    );
    let input_data: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
    let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();
    let full_output = linear(&input, &full_weight)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();

    let mut summed = vec![0.0_f32; n];
    for rank in 0..world_size {
        let k_shard = k / world_size;
        let k_start = rank * k_shard;

        let qw_s = slice_rows(&qw, n, k_start / 8, k_shard / 8, 4);
        let g_start = k_start / group_size;
        let g_shard = k_shard / group_size;
        let sc_s = slice_rows(&sc, n, g_start, g_shard, 2);
        let qz_s = slice_rows(&qz, n / 8, g_start, g_shard, 4);

        let shard_weight = LinearWeight::Quantized(
            QuantizedTensor::from_gptq_raw(
                &ctx,
                &[n, k_shard],
                DType::GPTQ_INT4,
                &qw_s,
                &sc_s,
                &qz_s,
                group_size,
            )
            .unwrap(),
        );

        let input_shard_data: Vec<f32> = input_data[k_start..k_start + k_shard].to_vec();
        let input_shard = CudaTensor::from_slice(&ctx, &[1, k_shard], &input_shard_data).unwrap();
        let shard_out = linear(&input_shard, &shard_weight)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        for (j, v) in shard_out.iter().enumerate() {
            summed[j] += v;
        }
    }

    for (i, (a, b)) in full_output.iter().zip(summed.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.5,
            "Row shard mismatch at {i}: full={a} vs summed={b}"
        );
    }
}
