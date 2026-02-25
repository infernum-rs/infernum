//! Procedural macros for Infernum's block fusion system.
//!
//! - [`define_block!`] — Wraps a function to support automatic fusion dispatch.
//! - [`define_fusion!`] — Registers a fused replacement for a named block.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Ident, ItemFn};

/// Wrap a function as a fusible block.
///
/// Generates two items:
/// 1. `{name}_decomposed` — the original function body, always available.
/// 2. `{name}` — a dispatcher that checks the fusion registry each call.
///
/// The function signature is preserved as-is (including any generics).
/// In debug builds (without `force-fuse`), the dispatcher always calls
/// the decomposed version. In release builds (without `no-fuse`), it
/// checks the registry via `fusion::get` and dispatches to the fused
/// version if one has been registered for the concrete type.
///
/// # Example
///
/// ```ignore
/// define_block! {
///     pub fn swiglu(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
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
    let fn_name_str = fn_name.to_string();
    let decomposed_name = format_ident!("{fn_name}_decomposed");

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

        // Dispatcher: checks fusion registry, falls back to decomposed
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

            // Release (or force-fuse): check the fusion registry
            match ::infernum::fusion::get::<fn(#(#param_types),*) -> #return_type>(#fn_name_str) {
                Some(f) => f(#(#param_forwards),*),
                None => #decomposed_name(#(#param_forwards),*),
            }
        }
    };

    output.into()
}

/// Input for `define_fusion!`: `name: "block_name", fn ...`
struct DefineFusionInput {
    block_name: syn::LitStr,
    func: ItemFn,
}

impl syn::parse::Parse for DefineFusionInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Parse `name: "block_name",`
        let name_kw: Ident = input.parse()?;
        if name_kw != "name" {
            return Err(syn::Error::new(name_kw.span(), "expected `name`"));
        }
        input.parse::<syn::Token![:]>()?;
        let block_name: syn::LitStr = input.parse()?;
        input.parse::<syn::Token![,]>()?;

        // Parse the function
        let func: ItemFn = input.parse()?;

        Ok(Self { block_name, func })
    }
}

/// Register a fused replacement for a block.
///
/// The fused function is emitted unchanged. An `inventory::submit!` call
/// is generated to register it with the fusion registry at startup
/// (when [`infernum::fusion::init`] is called).
///
/// # Example
///
/// ```ignore
/// define_fusion! {
///     name: "swiglu",
///     pub fn swiglu_fused(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
///         silu_mul_kernel(gate, up)
///     }
/// }
/// ```
#[proc_macro]
pub fn define_fusion(input: TokenStream) -> TokenStream {
    let DefineFusionInput { block_name, func } = parse_macro_input!(input as DefineFusionInput);

    let vis = &func.vis;
    let sig = &func.sig;
    let body = &func.block;
    let fn_name = &sig.ident;

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
        // The fused implementation (unchanged)
        #vis #sig #body

        // Register with the fusion registry via inventory
        ::inventory::submit! {
            ::infernum::fusion::FusionInit(|| {
                type __FusedFnPtr = fn(#(#param_types),*) -> #return_type;
                ::infernum::fusion::register::<__FusedFnPtr>(#block_name, #fn_name as __FusedFnPtr);
            })
        }
    };

    output.into()
}
