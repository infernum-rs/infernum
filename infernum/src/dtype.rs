//! Data types for tensor elements

use std::fmt;

/// Default block size for block-quantized formats (`Q8_0`, `Q4_0`)
pub const QUANTIZATION_BLOCK_SIZE: usize = 32;

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// Brain floating point (16-bit)
    BF16,
    /// 32-bit unsigned integer
    U32,
    /// 8-bit block-quantized integer (block size 32, one f16 scale per block)
    Q8_0,
    /// 4-bit block-quantized integer (block size 32, one f16 scale per block)
    Q4_0,
    /// 8-bit floating point (E4M3 format: 4 exponent, 3 mantissa bits)
    F8E4M3,
}

impl DType {
    /// Size of the dtype in bytes per element.
    ///
    /// # Panics
    /// Panics for block-quantized types (`Q8_0`, `Q4_0`) where per-element size
    /// is not meaningful. Use [`block_size_in_bytes`](Self::block_size_in_bytes) instead.
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 => 1,
            Self::Q8_0 | Self::Q4_0 => panic!(
                "Block-quantized types have no fixed per-element size; use block_size_in_bytes()"
            ),
        }
    }

    /// Size in bytes for one quantization block (`QUANTIZATION_BLOCK_SIZE` elements).
    ///
    /// - `Q8_0`: 32 × 1 byte (int8) + 2 bytes (f16 scale) = 34
    /// - `Q4_0`: 32 × 0.5 bytes (int4) + 2 bytes (f16 scale) = 18
    ///
    /// # Panics
    /// Panics for non-block-quantized types.
    #[must_use]
    pub const fn block_size_in_bytes(self) -> usize {
        match self {
            // 32 int8 values + 1 f16 scale
            Self::Q8_0 => QUANTIZATION_BLOCK_SIZE + 2,
            // 32 int4 values packed into 16 bytes + 1 f16 scale
            Self::Q4_0 => QUANTIZATION_BLOCK_SIZE / 2 + 2,
            _ => panic!("block_size_in_bytes() is only valid for block-quantized types"),
        }
    }

    /// Whether this dtype is a quantized format
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4_0 | Self::F8E4M3)
    }

    /// Whether this dtype is a block-quantized format (has scale factors)
    #[must_use]
    pub const fn is_block_quantized(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4_0)
    }

    /// Convert from safetensors dtype string
    #[must_use]
    pub fn from_safetensors(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "F8_E4M3" => Some(Self::F8E4M3),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::U32 => write!(f, "u32"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q4_0 => write!(f, "q4_0"),
            Self::F8E4M3 => write!(f, "f8e4m3"),
        }
    }
}

/// Trait for types that can be used as tensor elements
pub trait TensorDType: Copy + Clone + Default + Send + Sync + 'static {
    /// The corresponding `DType` enum value
    const DTYPE: DType;
}

impl TensorDType for f32 {
    const DTYPE: DType = DType::F32;
}

impl TensorDType for half::f16 {
    const DTYPE: DType = DType::F16;
}

impl TensorDType for half::bf16 {
    const DTYPE: DType = DType::BF16;
}

impl TensorDType for u32 {
    const DTYPE: DType = DType::U32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
        assert_eq!(DType::U32.size_in_bytes(), 4);
        assert_eq!(DType::F8E4M3.size_in_bytes(), 1);
    }

    #[test]
    #[should_panic(expected = "block_size_in_bytes")]
    fn test_dtype_size_in_bytes_q8_panics() {
        let _ = DType::Q8_0.size_in_bytes();
    }

    #[test]
    #[should_panic(expected = "block_size_in_bytes")]
    fn test_dtype_size_in_bytes_q4_panics() {
        let _ = DType::Q4_0.size_in_bytes();
    }

    #[test]
    fn test_dtype_block_size_in_bytes() {
        // Q8_0: 32 int8 values + 2 bytes f16 scale = 34
        assert_eq!(DType::Q8_0.block_size_in_bytes(), 34);
        // Q4_0: 16 bytes (32 int4 values) + 2 bytes f16 scale = 18
        assert_eq!(DType::Q4_0.block_size_in_bytes(), 18);
    }

    #[test]
    fn test_dtype_is_quantized() {
        assert!(!DType::F32.is_quantized());
        assert!(!DType::F16.is_quantized());
        assert!(!DType::BF16.is_quantized());
        assert!(!DType::U32.is_quantized());
        assert!(DType::Q8_0.is_quantized());
        assert!(DType::Q4_0.is_quantized());
        assert!(DType::F8E4M3.is_quantized());
    }

    #[test]
    fn test_dtype_is_block_quantized() {
        assert!(!DType::F32.is_block_quantized());
        assert!(!DType::F8E4M3.is_block_quantized());
        assert!(DType::Q8_0.is_block_quantized());
        assert!(DType::Q4_0.is_block_quantized());
    }

    #[test]
    fn test_dtype_from_safetensors() {
        assert_eq!(DType::from_safetensors("F32"), Some(DType::F32));
        assert_eq!(DType::from_safetensors("F16"), Some(DType::F16));
        assert_eq!(DType::from_safetensors("BF16"), Some(DType::BF16));
        assert_eq!(DType::from_safetensors("F8_E4M3"), Some(DType::F8E4M3));
        assert_eq!(DType::from_safetensors("I32"), None);
        assert_eq!(DType::from_safetensors("invalid"), None);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::F16), "f16");
        assert_eq!(format!("{}", DType::BF16), "bf16");
        assert_eq!(format!("{}", DType::U32), "u32");
        assert_eq!(format!("{}", DType::Q8_0), "q8_0");
        assert_eq!(format!("{}", DType::Q4_0), "q4_0");
        assert_eq!(format!("{}", DType::F8E4M3), "f8e4m3");
    }

    #[test]
    fn test_tensor_dtype_trait() {
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(half::f16::DTYPE, DType::F16);
        assert_eq!(half::bf16::DTYPE, DType::BF16);
        assert_eq!(u32::DTYPE, DType::U32);
    }
}
