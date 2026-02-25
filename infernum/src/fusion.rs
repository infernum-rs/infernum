//! Fusion registry for block-level kernel fusion.
//!
//! Provides a global registry mapping `(block_name, TypeId)` to fused
//! function pointers. This allows `define_block!` dispatchers to look up
//! a fused replacement for any concrete type at call time.
//!
//! - [`register`] — stores a fused function pointer for a block + type.
//! - [`get`] — retrieves it (returns `None` if no fusion is registered).
//! - [`init`] — runs all deferred registrations collected via `inventory`.

#![allow(clippy::missing_panics_doc)]

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

/// Key: (block name, function pointer `TypeId`).
type RegistryMap = HashMap<(&'static str, TypeId), Box<dyn Any + Send + Sync>>;

static REGISTRY: LazyLock<RwLock<RegistryMap>> = LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register a fused function pointer for a named block.
///
/// `T` is the concrete function pointer type, e.g.
/// `fn(&CudaTensor, &CudaTensor) -> Result<CudaTensor>`.
///
/// Subsequent calls to [`get::<T>(name)`](get) will return this pointer.
/// If a fusion for the same `(name, T)` is already registered, it is
/// silently replaced.
pub fn register<T: Any + Send + Sync + Copy>(name: &'static str, f: T) {
    let key = (name, TypeId::of::<T>());
    let mut map = REGISTRY.write().expect("fusion registry poisoned");
    map.insert(key, Box::new(f));
}

/// Look up a fused replacement for a named block.
///
/// Returns `Some(f)` if a fusion of type `T` was registered for `name`,
/// `None` otherwise. The lookup acquires a read lock on the registry —
/// negligible cost compared to a GPU kernel launch.
pub fn get<T: Any + Copy>(name: &'static str) -> Option<T> {
    let key = (name, TypeId::of::<T>());
    let map = REGISTRY.read().expect("fusion registry poisoned");
    map.get(&key)
        .and_then(|boxed| boxed.downcast_ref::<T>())
        .copied()
}

/// A deferred fusion initializer collected across crates at link time.
///
/// Each `define_fusion!` invocation submits one of these. The contained
/// closure calls [`register`] for the corresponding block.
pub struct FusionInit(pub fn());

inventory::collect!(FusionInit);

/// Run all fusion registrations.
///
/// Call once at startup before inference begins. After this returns,
/// every block that has a matching `define_fusion!` will dispatch to
/// its fused implementation (in release builds, or when `force-fuse`
/// is enabled).
///
/// Safe to call multiple times — subsequent calls just re-register
/// the same entries.
pub fn init() {
    for entry in inventory::iter::<FusionInit> {
        (entry.0)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        type AddFn = fn(i32, i32) -> i32;
        let f: AddFn = |a, b| a + b;
        register::<AddFn>("test_add", f);

        let retrieved = get::<AddFn>("test_add");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap()(2, 3), 5);
    }

    #[test]
    fn test_get_missing() {
        type MulFn = fn(i32, i32) -> i32;
        let result = get::<MulFn>("nonexistent_block");
        assert!(result.is_none());
    }

    #[test]
    fn test_type_mismatch_returns_none() {
        type AddFn = fn(i32, i32) -> i32;
        type OtherFn = fn(f32, f32) -> f32;

        let f: AddFn = |a, b| a + b;
        register::<AddFn>("test_typed", f);

        // Same name, different type → None
        assert!(get::<OtherFn>("test_typed").is_none());
        // Same name, same type → Some
        assert!(get::<AddFn>("test_typed").is_some());
    }

    #[test]
    fn test_init_runs_inventory() {
        init();
        // Fusions registered via define_fusion! in other modules should
        // now be available. This test just verifies init() doesn't panic.
    }
}
