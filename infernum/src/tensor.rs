//! Tensor trait definition

use crate::dtype::DType;

/// Core tensor trait that defines the interface for all tensor implementations
///
/// Different backends (CUDA, Metal, CPU) implement this trait to provide
/// hardware-specific tensor operations while maintaining a unified interface.
pub trait Tensor: Sized {
    /// Returns the shape of the tensor as a slice of dimensions
    fn shape(&self) -> &[usize];

    /// Returns the data type of tensor elements
    fn dtype(&self) -> DType;

    /// Create a view with a different shape (same data, same number of elements).
    ///
    /// # Panics
    /// Panics if the new shape has a different number of elements.
    #[must_use]
    fn reshape(&self, shape: &[usize]) -> Self;

    /// Create a zero-copy sub-slice view starting at element `offset` with the
    /// given `shape`.
    ///
    /// # Panics
    /// Panics if the view extends beyond the backing allocation.
    #[must_use]
    fn slice_view(&self, offset: usize, shape: &[usize]) -> Self;

    /// Returns the total number of elements in the tensor
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns the number of dimensions (rank) of the tensor
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Returns the stride for each dimension
    fn strides(&self) -> Vec<usize> {
        let shape = self.shape();
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Returns true if the tensor is contiguous in memory
    fn is_contiguous(&self) -> bool {
        true // Default: assume contiguous
    }

    /// Returns the size of the tensor data in bytes
    fn size_in_bytes(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[derive(Clone)]
    struct FakeTensor {
        shape: Vec<usize>,
        dtype: DType,
    }

    impl Tensor for FakeTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn dtype(&self) -> DType {
            self.dtype
        }

        fn reshape(&self, shape: &[usize]) -> Self {
            let new_numel: usize = shape.iter().product();
            assert_eq!(self.numel(), new_numel);
            Self {
                shape: shape.to_vec(),
                dtype: self.dtype,
            }
        }

        fn slice_view(&self, _offset: usize, shape: &[usize]) -> Self {
            Self {
                shape: shape.to_vec(),
                dtype: self.dtype,
            }
        }
    }

    #[test]
    fn test_numel() {
        let t = FakeTensor {
            shape: vec![2, 3, 4],
            dtype: DType::F32,
        };
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_numel_scalar() {
        let t = FakeTensor {
            shape: vec![],
            dtype: DType::F32,
        };
        assert_eq!(t.numel(), 1);
    }

    #[test]
    fn test_ndim() {
        let t = FakeTensor {
            shape: vec![2, 3],
            dtype: DType::F16,
        };
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn test_strides_3d() {
        let t = FakeTensor {
            shape: vec![2, 3, 4],
            dtype: DType::F32,
        };
        assert_eq!(t.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_strides_1d() {
        let t = FakeTensor {
            shape: vec![5],
            dtype: DType::F32,
        };
        assert_eq!(t.strides(), vec![1]);
    }

    #[test]
    fn test_strides_empty() {
        let t = FakeTensor {
            shape: vec![],
            dtype: DType::F32,
        };
        assert_eq!(t.strides(), Vec::<usize>::new());
    }

    #[test]
    fn test_is_contiguous_default() {
        let t = FakeTensor {
            shape: vec![2, 3],
            dtype: DType::F32,
        };
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_size_in_bytes_f32() {
        let t = FakeTensor {
            shape: vec![2, 3],
            dtype: DType::F32,
        };
        assert_eq!(t.size_in_bytes(), 24); // 6 * 4
    }

    #[test]
    fn test_size_in_bytes_f16() {
        let t = FakeTensor {
            shape: vec![2, 3],
            dtype: DType::F16,
        };
        assert_eq!(t.size_in_bytes(), 12); // 6 * 2
    }

    #[test]
    fn test_size_in_bytes_bf16() {
        let t = FakeTensor {
            shape: vec![4],
            dtype: DType::BF16,
        };
        assert_eq!(t.size_in_bytes(), 8); // 4 * 2
    }
}
