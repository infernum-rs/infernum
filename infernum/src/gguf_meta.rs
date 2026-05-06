//! GGUF metadata value type
//!
//! This is the pure-data enum representing parsed GGUF metadata values.
//! It lives in the core crate so that non-CUDA code (e.g., `GgufTokenizer`)
//! can use it without depending on `infernum-cuda`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

/// A parsed GGUF metadata value
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Try to interpret the value as a usize
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::U8(v) => Some(*v as usize),
            Self::I8(v) => usize::try_from(*v).ok(),
            Self::U16(v) => Some(*v as usize),
            Self::I16(v) => usize::try_from(*v).ok(),
            Self::U32(v) => Some(*v as usize),
            Self::I32(v) => usize::try_from(*v).ok(),
            Self::U64(v) => usize::try_from(*v).ok(),
            Self::I64(v) => usize::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to interpret the value as an f32
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to interpret the value as a string
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to interpret the value as a bool
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Extract a required `usize` value from GGUF metadata by key.
///
/// # Errors
///
/// Returns [`crate::Error::InvalidShape`] if the key is absent or the value
/// cannot be interpreted as a `usize`.
pub fn get_usize(
    metadata: &std::collections::HashMap<String, GgufValue>,
    key: &str,
) -> crate::Result<usize> {
    metadata
        .get(key)
        .and_then(GgufValue::as_usize)
        .ok_or_else(|| crate::Error::InvalidShape(format!("Missing GGUF metadata: {key}")))
}

/// Extract an optional `f32` value from GGUF metadata, returning `default` if absent.
#[must_use]
pub fn get_f32(
    metadata: &std::collections::HashMap<String, GgufValue>,
    key: &str,
    default: f32,
) -> f32 {
    metadata
        .get(key)
        .and_then(GgufValue::as_f32)
        .unwrap_or(default)
}
