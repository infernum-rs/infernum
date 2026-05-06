//! Shared serde helper functions for model config deserialization.

use serde::Deserialize;

/// Deserialize a field that may be a single `u32` or an array of `u32`.
///
/// When an array is provided, the first element is used. Intended for fields
/// like `eos_token_id` that some model configs specify as a scalar and others
/// as a list.
///
/// # Errors
///
/// Returns a deserialization error if the value is an array but empty.
pub fn deserialize_u32_or_first<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SingleOrVec {
        Single(u32),
        Vec(Vec<u32>),
    }

    match SingleOrVec::deserialize(deserializer)? {
        SingleOrVec::Single(v) => Ok(v),
        SingleOrVec::Vec(v) => v
            .first()
            .copied()
            .ok_or_else(|| serde::de::Error::custom("eos_token_id array is empty")),
    }
}
