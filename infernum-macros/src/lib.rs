//! Procedural macros for Infernum's block fusion system.
//!
//! - [`define_block!`] — Wraps a function to support automatic fusion dispatch.
//! - [`define_fusion!`] — Registers a fused replacement for a named block.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Ident, ItemFn, Path};

/// Generate an `UPPER_SNAKE_CASE` `OnceLock` static name from a function name.
///
/// `attention_kv` → `ATTENTION_KV_FUSED`
fn fused_static_name(fn_name: &Ident) -> Ident {
    let upper = fn_name.to_string().to_uppercase();
    format_ident!("{upper}_FUSED")
}

/// Wrap a function as a fusible block.
///
/// Generates three items:
/// 1. `{name}_decomposed` — the original function body, always available.
/// 2. `{NAME}_FUSED` — a `OnceLock` static that fusion rules populate.
/// 3. `{name}` — a dispatcher that checks for a fused replacement.
///
/// In debug builds (without `force-fuse`), the dispatcher always calls
/// the decomposed version. In release builds (without `no-fuse`), it
/// checks the `OnceLock` and dispatches to the fused version if set.
///
/// # Example
///
/// ```ignore
/// define_block! {
///     pub fn swiglu(gate: &CudaTensor<f32>, up: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
///         let activated = silu(gate)?;
///         mul(&activated, up)
///     }
/// }
/// ```
#[proc_macro]
pub fn define_block(input: TokenStream) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let body = &func.block;
    let fn_name = &sig.ident;
    let decomposed_name = format_ident!("{fn_name}_decomposed");
    let static_name = fused_static_name(fn_name);

    // Build the decomposed function signature (same as original, renamed)
    let mut decomposed_sig = sig.clone();
    decomposed_sig.ident = decomposed_name.clone();

    // Extract parameter names for forwarding in the dispatcher
    let param_forwards: Vec<_> = sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(pat_type) => {
                let pat = &pat_type.pat;
                quote! { #pat }
            }
            syn::FnArg::Receiver(_) => quote! { self },
        })
        .collect();

    // Build the function pointer type from the signature
    let param_types: Vec<_> = sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(pat_type) => {
                let ty = &pat_type.ty;
                quote! { #ty }
            }
            syn::FnArg::Receiver(_) => quote! { Self },
        })
        .collect();

    let return_type = match &sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };

    let output = quote! {
        // The decomposed (original) implementation
        #[allow(dead_code)]
        #vis #decomposed_sig #body

        // Static slot for a fused replacement — populated by `define_fusion!`
        #vis static #static_name: ::std::sync::OnceLock<fn(#(#param_types),*) -> #return_type> =
            ::std::sync::OnceLock::new();

        // Dispatcher: checks for fused replacement, falls back to decomposed
        #(#attrs)*
        #vis #sig {
            // no-fuse: always decomposed
            if cfg!(feature = "no-fuse") {
                return #decomposed_name(#(#param_forwards),*);
            }

            // Debug without force-fuse: always decomposed
            if cfg!(debug_assertions) && !cfg!(feature = "force-fuse") {
                return #decomposed_name(#(#param_forwards),*);
            }

            // Release (or force-fuse): check for fused replacement
            match #static_name.get() {
                Some(f) => f(#(#param_forwards),*),
                None => #decomposed_name(#(#param_forwards),*),
            }
        }
    };

    output.into()
}

/// Input for `define_fusion!`: `block: PATH, fn ...`
struct DefineFusionInput {
    block_path: Path,
    func: ItemFn,
}

impl syn::parse::Parse for DefineFusionInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Parse `block: some::path::STATIC_NAME,`
        let block_kw: Ident = input.parse()?;
        if block_kw != "block" {
            return Err(syn::Error::new(block_kw.span(), "expected `block`"));
        }
        input.parse::<syn::Token![:]>()?;
        let block_path: Path = input.parse()?;
        input.parse::<syn::Token![,]>()?;

        // Parse the function
        let func: ItemFn = input.parse()?;

        Ok(Self { block_path, func })
    }
}

/// Register a fused replacement for a block.
///
/// The fused function is emitted unchanged. An `inventory::submit!` call
/// is generated to populate the block's `OnceLock` static at startup
/// (when [`infernum::fusion::init`] is called).
///
/// # Example
///
/// ```ignore
/// define_fusion! {
///     block: super::SWIGLU_FUSED,
///     pub fn swiglu_fused(gate: &CudaTensor<f32>, up: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
///         silu_mul_kernel(gate, up)
///     }
/// }
/// ```
#[proc_macro]
pub fn define_fusion(input: TokenStream) -> TokenStream {
    let DefineFusionInput { block_path, func } = parse_macro_input!(input as DefineFusionInput);

    let vis = &func.vis;
    let sig = &func.sig;
    let body = &func.block;
    let fn_name = &sig.ident;

    let output = quote! {
        // The fused implementation (unchanged)
        #vis #sig #body

        // Register with the fusion system via inventory
        ::inventory::submit! {
            ::infernum::fusion::FusionInit(|| {
                let _ = #block_path.set(#fn_name as _);
            })
        }
    };

    output.into()
}
