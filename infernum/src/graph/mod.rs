//! Computation graph representation for LLM inference.
//!
//! The graph is built by model code and consumed by backend executors.
//! It captures the full forward pass as a DAG of typed operations,
//! enabling fusion, scheduling, and memory planning.

mod builder;
mod node;
mod ops;

pub use builder::Graph;
pub use node::{GraphNode, NodeId, WeightId, WeightMeta, WeightRef};
pub use ops::{MoeExpertIds, Op};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::dtype::DType;

    /// Minimal backend for testing. Not a real backend — just satisfies
    /// the trait bounds so we can instantiate `Graph<TestBackend>`.
    struct TestBackend;

    #[derive(Clone)]
    struct DummyTensor;

    impl crate::tensor::Tensor for DummyTensor {
        fn shape(&self) -> &[usize] {
            &[]
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn reshape(&self, _shape: &[usize]) -> Self {
            Self
        }
        fn slice_view(&self, _offset: usize, _shape: &[usize]) -> Self {
            Self
        }
    }

    struct DummyLogits;

    impl crate::logits::Logits for DummyLogits {
        fn vocab_size(&self) -> usize {
            0
        }
        fn batch_size(&self) -> usize {
            0
        }
        fn argmax(&self, _batch_index: usize) -> crate::Result<u32> {
            Ok(0)
        }
        fn sample_top_p(
            &self,
            _batch_index: usize,
            _temperature: f32,
            _top_p: f32,
            _rng_seed: u64,
            _repetition_penalty: f32,
            _recent_tokens: &[u32],
        ) -> crate::Result<u32> {
            Ok(0)
        }
    }

    struct DummyRuntimeState;

    impl crate::runtime_state::RuntimeStateInit for DummyRuntimeState {
        fn new(
            _batch_config: &crate::runtime_state::BatchConfig,
            _block_config: &crate::block_allocator::BlockConfig,
        ) -> crate::Result<Self> {
            Ok(Self)
        }

        fn new_placeholder() -> Self {
            Self
        }
    }

    impl Backend for TestBackend {
        type Tensor = DummyTensor;
        type DeviceHandle = ();
        type PagedKvCache = ();
        type KvCache = ();
        type RuntimeState = DummyRuntimeState;
        type Logits = DummyLogits;
        type Comm = ();

        fn logits_from_tensor(_tensor: Self::Tensor) -> Self::Logits {
            DummyLogits
        }
    }

    #[test]
    fn empty_graph() {
        let graph = Graph::<TestBackend>::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn input_reshape_output() {
        let mut graph = Graph::<TestBackend>::new();

        // Input: (4, 128)
        let input = graph.add_input(&[4, 128], DType::BF16);
        assert_eq!(graph.node_shape(input), &[4, 128]);
        assert_eq!(graph.node_dtype(input), DType::BF16);

        // Reshape to (2, 256)
        let reshaped = graph.push_node(
            Op::Reshape {
                shape: smallvec::smallvec![2, 256],
            },
            &[input],
            vec![2, 256],
            DType::BF16,
        );

        graph.set_output(reshaped);

        assert_eq!(graph.len(), 2);
        assert!(!graph.is_empty());
        assert_eq!(graph.node_shape(reshaped), &[2, 256]);

        // Verify the output was recorded.
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0], reshaped);
    }

    #[test]
    fn register_weights_and_retrieve_metadata() {
        let mut graph = Graph::<TestBackend>::new();

        let tw = graph.register_tensor_weight("ln.weight", &[4096], DType::BF16);
        let lw = graph.register_linear_weight("attn.q_proj", &[4096, 4096], DType::BF16);

        let tmeta = graph.tensor_weight_meta(tw);
        assert_eq!(tmeta.name, "ln.weight");
        assert_eq!(tmeta.shape, vec![4096]);
        assert_eq!(tmeta.dtype, DType::BF16);

        let lmeta = graph.linear_weight_meta(lw);
        assert_eq!(lmeta.name, "attn.q_proj");
        assert_eq!(lmeta.shape, vec![4096, 4096]);
        assert_eq!(lmeta.dtype, DType::BF16);
    }

    #[test]
    fn mini_transformer_connectivity() {
        let mut graph = Graph::<TestBackend>::new();

        // Weights
        let norm_w = graph.register_tensor_weight("ln.weight", &[512], DType::BF16);
        let proj_w = graph.register_linear_weight("proj.weight", &[512, 512], DType::BF16);

        // Input: (8, 512)
        let input = graph.add_input(&[8, 512], DType::BF16);

        // RmsNorm
        let normed = graph.push_node(
            Op::RmsNorm {
                weight: norm_w,
                eps: 1e-5,
            },
            &[input],
            vec![8, 512],
            DType::BF16,
        );

        // Linear
        let projected = graph.push_node(
            Op::Linear { weight: proj_w },
            &[normed],
            vec![8, 512],
            DType::BF16,
        );

        // Residual add
        let added = graph.push_node(Op::Add, &[input, projected], vec![8, 512], DType::BF16);

        graph.set_output(added);

        assert_eq!(graph.len(), 4); // input, norm, linear, add
        assert_eq!(graph.node_shape(added), &[8, 512]);

        // Verify connectivity: the add node should have 2 inputs.
        let add_node = &graph.nodes[added.0 as usize];
        assert_eq!(add_node.inputs.len(), 2);
        assert_eq!(add_node.inputs[0], input);
        assert_eq!(add_node.inputs[1], projected);
    }

    #[test]
    fn multi_output_add_rmsnorm() {
        let mut graph = Graph::<TestBackend>::new();

        let norm_w = graph.register_tensor_weight("ln.weight", &[256], DType::F32);

        let residual = graph.add_input(&[4, 256], DType::F32);
        let delta = graph.add_input(&[4, 256], DType::F32);

        let (updated_residual, normalized) = graph.push_node_pair(
            Op::AddRmsNorm {
                weight: norm_w,
                eps: 1e-6,
            },
            &[residual, delta],
            vec![4, 256],
            DType::F32,
            vec![4, 256],
            DType::F32,
        );

        assert_eq!(graph.len(), 4); // 2 inputs + primary + secondary
        assert_eq!(graph.node_shape(updated_residual), &[4, 256]);
        assert_eq!(graph.node_shape(normalized), &[4, 256]);
        assert_eq!(graph.node_dtype(updated_residual), DType::F32);
        assert_eq!(graph.node_dtype(normalized), DType::F32);

        // The secondary node should be a SecondOutput referencing the primary.
        let secondary_node = &graph.nodes[normalized.0 as usize];
        match &secondary_node.op {
            Op::SecondOutput { source } => assert_eq!(*source, updated_residual),
            other => panic!("Expected SecondOutput, got {other:?}"),
        }
    }

    #[test]
    fn set_output_records_multiple() {
        let mut graph = Graph::<TestBackend>::new();

        let a = graph.add_input(&[1, 128], DType::F32);
        let b = graph.add_input(&[1, 128], DType::F32);

        graph.set_output(a);
        graph.set_output(b);

        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.outputs[0], a);
        assert_eq!(graph.outputs[1], b);
    }

    #[test]
    fn weight_ref_variants() {
        let tid = WeightId(0);
        let lid = WeightId(1);

        let tref = WeightRef::Tensor(tid);
        let lref = WeightRef::Linear(lid);

        // Verify equality and debug formatting work.
        assert_eq!(tref, WeightRef::Tensor(WeightId(0)));
        assert_ne!(tref, lref);
        assert_eq!(format!("{tref:?}"), "Tensor(WeightId(0))");
        assert_eq!(format!("{lref:?}"), "Linear(WeightId(1))");
    }
}
