//! Flat byte arena for graph execution.
//!
//! The executor allocates a single `Arena` and uses the memory planner's
//! offset assignments to carve out typed slices for each node's output.
//! This avoids per-op allocation and gives the CPU cache-friendly, contiguous
//! memory for all intermediate activations.

use std::mem;

/// A flat byte arena for graph execution.
///
/// The memory planner assigns each graph node an `(offset, size)` pair within
/// this arena. The executor uses those offsets to obtain typed slices for
/// reading inputs and writing outputs.
///
/// # Alignment
///
/// The arena is backed by a `Vec<u8>`. Callers are responsible for ensuring
/// that offsets passed to [`f32_slice`](Arena::f32_slice) and
/// [`f32_slice_mut`](Arena::f32_slice_mut) are 4-byte aligned (the memory
/// planner guarantees this because all `DType` sizes are powers of two and
/// offsets are computed from cumulative byte sizes).
pub struct Arena {
    data: Vec<u8>,
}

impl Arena {
    /// Create a new arena of the given size in bytes, zero-initialized.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Get an immutable `f32` slice at the given byte offset and element count.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `offset + len * 4` exceeds the arena size
    /// - `offset` is not 4-byte aligned
    #[must_use]
    pub fn f32_slice(&self, offset: usize, len: usize) -> &[f32] {
        let byte_len = len * mem::size_of::<f32>();
        let end = offset + byte_len;
        assert!(
            end <= self.data.len(),
            "slice [{offset}..{end}) exceeds arena size {}",
            self.data.len()
        );
        bytemuck::cast_slice(&self.data[offset..end])
    }

    /// Get a mutable `f32` slice at the given byte offset and element count.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `offset + len * 4` exceeds the arena size
    /// - `offset` is not 4-byte aligned
    #[must_use]
    pub fn f32_slice_mut(&mut self, offset: usize, len: usize) -> &mut [f32] {
        let byte_len = len * mem::size_of::<f32>();
        let end = offset + byte_len;
        assert!(
            end <= self.data.len(),
            "slice [{offset}..{end}) exceeds arena size {}",
            self.data.len()
        );
        bytemuck::cast_slice_mut(&mut self.data[offset..end])
    }

    /// Total arena size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Borrow two non-overlapping mutable `f32` slices simultaneously.
    ///
    /// This is needed for ops like `AddRmsNorm` that write two separate
    /// outputs, or for in-place ops where one input is also the output.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The two byte ranges overlap
    /// - Either range exceeds the arena bounds
    /// - Either offset is not 4-byte aligned
    #[must_use]
    pub fn f32_slice_pair_mut(
        &mut self,
        offset1: usize,
        len1: usize,
        offset2: usize,
        len2: usize,
    ) -> (&mut [f32], &mut [f32]) {
        let byte_len1 = len1 * mem::size_of::<f32>();
        let byte_len2 = len2 * mem::size_of::<f32>();
        let end1 = offset1 + byte_len1;
        let end2 = offset2 + byte_len2;

        assert!(
            end1 <= self.data.len(),
            "first slice [{offset1}..{end1}) exceeds arena size {}",
            self.data.len()
        );
        assert!(
            end2 <= self.data.len(),
            "second slice [{offset2}..{end2}) exceeds arena size {}",
            self.data.len()
        );

        // Ensure non-overlapping ranges.
        assert!(
            end1 <= offset2 || end2 <= offset1,
            "slices [{offset1}..{end1}) and [{offset2}..{end2}) overlap"
        );

        // SAFETY: we verified the two byte ranges are non-overlapping and
        // within bounds. `bytemuck::cast_slice_mut` will verify alignment.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let slice1 = std::slice::from_raw_parts_mut(ptr.add(offset1), byte_len1);
            let slice2 = std::slice::from_raw_parts_mut(ptr.add(offset2), byte_len2);
            (
                bytemuck::cast_slice_mut(slice1),
                bytemuck::cast_slice_mut(slice2),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_and_read_f32() {
        let mut arena = Arena::new(32); // 8 f32s

        let out = arena.f32_slice_mut(0, 4);
        out[0] = 1.0;
        out[1] = 2.0;
        out[2] = 3.0;
        out[3] = 4.0;

        let read = arena.f32_slice(0, 4);
        assert_eq!(read, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn write_at_offset() {
        let mut arena = Arena::new(32);

        // Write at byte offset 16 (element offset 4).
        let out = arena.f32_slice_mut(16, 2);
        out[0] = 10.0;
        out[1] = 20.0;

        // First 4 elements should still be zero.
        let first = arena.f32_slice(0, 4);
        assert_eq!(first, &[0.0, 0.0, 0.0, 0.0]);

        // Read back at offset.
        let second = arena.f32_slice(16, 2);
        assert_eq!(second, &[10.0, 20.0]);
    }

    #[test]
    fn non_overlapping_pair_mut() {
        let mut arena = Arena::new(32); // 8 f32s

        let (a, b) = arena.f32_slice_pair_mut(0, 4, 16, 4);

        a[0] = 1.0;
        a[3] = 4.0;
        b[0] = 5.0;
        b[3] = 8.0;

        // Read back independently.
        let first = arena.f32_slice(0, 4);
        assert_eq!(first[0], 1.0);
        assert_eq!(first[3], 4.0);

        let second = arena.f32_slice(16, 4);
        assert_eq!(second[0], 5.0);
        assert_eq!(second[3], 8.0);
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn overlapping_pair_panics() {
        let mut arena = Arena::new(32);
        // Ranges [0..16) and [8..24) overlap.
        let _ = arena.f32_slice_pair_mut(0, 4, 8, 4);
    }

    #[test]
    #[should_panic(expected = "exceeds arena size")]
    fn out_of_bounds_panics() {
        let mut arena = Arena::new(16);
        // 8 f32s = 32 bytes, but arena is only 16 bytes.
        let _ = arena.f32_slice_mut(0, 8);
    }

    #[test]
    fn zero_length_slices() {
        let arena = Arena::new(16);
        let empty = arena.f32_slice(0, 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn arena_size() {
        let arena = Arena::new(1024);
        assert_eq!(arena.size(), 1024);
    }

    #[test]
    fn new_arena_is_zeroed() {
        let arena = Arena::new(16);
        let data = arena.f32_slice(0, 4);
        assert_eq!(data, &[0.0, 0.0, 0.0, 0.0]);
    }
}
