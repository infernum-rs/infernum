//! Generic weight store for graph execution.
//!
//! Maps [`WeightId`] indices to loaded weight values. The graph builder
//! registers weights by name and shape, producing `WeightId` handles.
//! During model loading, actual weight data is pushed into the store in
//! the same order, so `WeightId(n)` corresponds to the `n`-th tensor
//! or linear weight.

use super::node::WeightId;

/// Generic weight store parameterized by tensor weight (`TW`) and linear
/// weight (`LW`) types.
///
/// For `CpuBackend`, `TW` and `LW` might both be `CpuTensor`.
/// For `CudaBackend`, `TW = CudaTensor` and `LW = CudaTensor` (or a
/// quantized wrapper).
pub struct WeightStore<TW, LW> {
    tensor_weights: Vec<TW>,
    linear_weights: Vec<LW>,
}

impl<TW, LW> WeightStore<TW, LW> {
    /// Create an empty weight store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensor_weights: Vec::new(),
            linear_weights: Vec::new(),
        }
    }

    /// Create an empty weight store with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(tensor_capacity: usize, linear_capacity: usize) -> Self {
        Self {
            tensor_weights: Vec::with_capacity(tensor_capacity),
            linear_weights: Vec::with_capacity(linear_capacity),
        }
    }

    /// Push a tensor weight (embedding table, layernorm, `RoPE` cache, bias).
    /// Returns its index for verification against the expected `WeightId`.
    pub fn push_tensor_weight(&mut self, weight: TW) -> usize {
        let idx = self.tensor_weights.len();
        self.tensor_weights.push(weight);
        idx
    }

    /// Push a linear weight (dense or quantized matmul weight).
    /// Returns its index for verification against the expected `WeightId`.
    pub fn push_linear_weight(&mut self, weight: LW) -> usize {
        let idx = self.linear_weights.len();
        self.linear_weights.push(weight);
        idx
    }

    /// Get a tensor weight by `WeightId`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[must_use]
    pub fn tensor_weight(&self, id: WeightId) -> &TW {
        &self.tensor_weights[id.0 as usize]
    }

    /// Get a linear weight by `WeightId`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[must_use]
    pub fn linear_weight(&self, id: WeightId) -> &LW {
        &self.linear_weights[id.0 as usize]
    }

    /// Number of tensor weights in the store.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensor_weights.len()
    }

    /// Number of linear weights in the store.
    #[must_use]
    pub fn linear_count(&self) -> usize {
        self.linear_weights.len()
    }
}

impl<TW, LW> Default for WeightStore<TW, LW> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_retrieve_tensor_weights() {
        let mut store = WeightStore::<Vec<f32>, Vec<f32>>::new();

        let idx0 = store.push_tensor_weight(vec![1.0, 2.0, 3.0]);
        let idx1 = store.push_tensor_weight(vec![4.0, 5.0]);

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(store.tensor_count(), 2);

        assert_eq!(store.tensor_weight(WeightId(0)), &[1.0, 2.0, 3.0]);
        assert_eq!(store.tensor_weight(WeightId(1)), &[4.0, 5.0]);
    }

    #[test]
    fn push_and_retrieve_linear_weights() {
        let mut store = WeightStore::<String, Vec<f32>>::new();

        let idx0 = store.push_linear_weight(vec![10.0, 20.0]);
        let idx1 = store.push_linear_weight(vec![30.0]);

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(store.linear_count(), 2);

        assert_eq!(store.linear_weight(WeightId(0)), &[10.0, 20.0]);
        assert_eq!(store.linear_weight(WeightId(1)), &[30.0]);
    }

    #[test]
    fn with_capacity_works() {
        let store = WeightStore::<f32, f32>::with_capacity(10, 20);
        assert_eq!(store.tensor_count(), 0);
        assert_eq!(store.linear_count(), 0);
    }

    #[test]
    fn default_is_empty() {
        let store = WeightStore::<f32, f32>::default();
        assert_eq!(store.tensor_count(), 0);
        assert_eq!(store.linear_count(), 0);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_tensor_panics() {
        let store = WeightStore::<f32, f32>::new();
        let _ = store.tensor_weight(WeightId(0));
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_linear_panics() {
        let store = WeightStore::<f32, f32>::new();
        let _ = store.linear_weight(WeightId(0));
    }
}
