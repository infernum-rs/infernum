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
