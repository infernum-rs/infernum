#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::module_name_repetitions,
    clippy::manual_div_ceil
)]

use std::marker::PhantomData;

use infernum::dtype::TensorDType;

use crate::config::GemmaConfig;

/// Gemma model supporting both Gemma 2 and Gemma 3 text architectures.
pub struct GemmaModel<T: TensorDType> {
    _phantom: PhantomData<T>,
    /// Model configuration.
    pub config: GemmaConfig,
}
