//! Fusion registry for block-level kernel fusion.
//!
//! `define_block!` generates a `OnceLock`-backed static for each block.
//! `define_fusion!` registers a fused replacement via `inventory`.
//! Calling [`init`] populates all statics so that block dispatchers
//! can resolve to their fused versions.

/// A deferred fusion initializer collected across crates at link time.
///
/// Each `define_fusion!` invocation submits one of these. The contained
/// closure calls `OnceLock::set` on the corresponding block's static.
pub struct FusionInit(pub fn());

inventory::collect!(FusionInit);

/// Run all fusion registrations.
///
/// Call once at startup before inference begins. After this returns,
/// every block that has a matching `define_fusion!` will dispatch to
/// its fused implementation (in release builds, or when `force-fuse`
/// is enabled).
///
/// Safe to call multiple times — subsequent calls are no-ops because
/// the underlying `OnceLock`s ignore duplicate `set` attempts.
pub fn init() {
    for entry in inventory::iter::<FusionInit> {
        (entry.0)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum_macros::{define_block, define_fusion};

    define_block! {
        fn add_values(a: i32, b: i32) -> i32 {
            a + b
        }
    }

    define_fusion! {
        block: ADD_VALUES_FUSED,
        fn add_values_fused(a: i32, b: i32) -> i32 {
            (a + b) * 100
        }
    }

    #[test]
    fn test_decomposed_always_available() {
        assert_eq!(add_values_decomposed(2, 3), 5);
    }

    #[test]
    #[cfg(not(feature = "force-fuse"))]
    fn test_debug_uses_decomposed() {
        // In debug builds (default for tests), the dispatcher
        // always calls decomposed — it never checks the OnceLock.
        assert_eq!(add_values(2, 3), 5);
    }

    #[test]
    fn test_fusion_init_populates_static() {
        init();
        // After init, the static should be populated
        assert!(ADD_VALUES_FUSED.get().is_some());
    }

    #[test]
    fn test_fused_function_directly() {
        assert_eq!(add_values_fused(2, 3), 500);
    }

    #[test]
    #[cfg(feature = "force-fuse")]
    fn test_force_fuse_dispatches_to_fused() {
        init();
        // With force-fuse, even in debug builds the dispatcher checks
        // the OnceLock and dispatches to the fused version.
        assert_eq!(add_values(2, 3), 500);
    }

    #[test]
    #[cfg(feature = "no-fuse")]
    fn test_no_fuse_always_decomposed() {
        init();
        // With no-fuse, even after init() the dispatcher always
        // returns the decomposed result.
        assert_eq!(add_values(2, 3), 5);
    }
}
