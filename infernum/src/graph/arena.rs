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

    /// Returns a raw mutable pointer to the start of the arena's byte buffer.
    ///
    /// # Safety
    ///
    /// Callers must ensure that any slices created from this pointer are
    /// disjoint and that alignment requirements are met.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
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

    /// Get an immutable `u32` slice at the given byte offset and element count.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `offset + len * 4` exceeds the arena size
    /// - `offset` is not 4-byte aligned
    #[must_use]
    pub fn u32_slice(&self, offset: usize, len: usize) -> &[u32] {
        let byte_len = len * mem::size_of::<u32>();
        let end = offset + byte_len;
        assert!(
            end <= self.data.len(),
            "slice [{offset}..{end}) exceeds arena size {}",
            self.data.len()
        );
        bytemuck::cast_slice(&self.data[offset..end])
    }

    /// Get a mutable `u32` slice at the given byte offset and element count.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `offset + len * 4` exceeds the arena size
    /// - `offset` is not 4-byte aligned
    #[must_use]
    pub fn u32_slice_mut(&mut self, offset: usize, len: usize) -> &mut [u32] {
        let byte_len = len * mem::size_of::<u32>();
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

    /// Borrow two immutable `f32` input slices and one mutable `f32` output
    /// slice simultaneously.
    ///
    /// Useful for element-wise binary ops (`add`, `mul`, `swiglu`) where two
    /// read-only inputs produce one written output.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Any byte range exceeds the arena bounds
    /// - The output range overlaps with either input range
    /// - Any offset is not 4-byte aligned
    #[must_use]
    pub fn two_slices_in_one_out(
        &mut self,
        in1_offset: usize,
        in1_len: usize,
        in2_offset: usize,
        in2_len: usize,
        out_offset: usize,
        out_len: usize,
    ) -> (&[f32], &[f32], &mut [f32]) {
        let in1_bytes = in1_len * mem::size_of::<f32>();
        let in2_bytes = in2_len * mem::size_of::<f32>();
        let out_bytes = out_len * mem::size_of::<f32>();
        let in1_end = in1_offset + in1_bytes;
        let in2_end = in2_offset + in2_bytes;
        let out_end = out_offset + out_bytes;
        let arena_len = self.data.len();

        assert!(
            in1_end <= arena_len,
            "in1 [{in1_offset}..{in1_end}) exceeds arena size {arena_len}"
        );
        assert!(
            in2_end <= arena_len,
            "in2 [{in2_offset}..{in2_end}) exceeds arena size {arena_len}"
        );
        assert!(
            out_end <= arena_len,
            "out [{out_offset}..{out_end}) exceeds arena size {arena_len}"
        );
        // The mutable output must not overlap with either immutable input.
        assert!(
            out_end <= in1_offset || in1_end <= out_offset,
            "out [{out_offset}..{out_end}) overlaps in1 [{in1_offset}..{in1_end})"
        );
        assert!(
            out_end <= in2_offset || in2_end <= out_offset,
            "out [{out_offset}..{out_end}) overlaps in2 [{in2_offset}..{in2_end})"
        );

        // SAFETY: `in1` and `in2` are cast to `*const` so they carry no
        // exclusive-access guarantee. `out` is the sole `*mut` region, and we
        // verified above that it does not overlap with either input.
        // All three ranges are within bounds.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let in1_raw = std::slice::from_raw_parts(ptr.add(in1_offset).cast_const(), in1_bytes);
            let in2_raw = std::slice::from_raw_parts(ptr.add(in2_offset).cast_const(), in2_bytes);
            let out_raw = std::slice::from_raw_parts_mut(ptr.add(out_offset), out_bytes);
            (
                bytemuck::cast_slice(in1_raw),
                bytemuck::cast_slice(in2_raw),
                bytemuck::cast_slice_mut(out_raw),
            )
        }
    }

    /// Borrow one immutable `f32` input slice and two non-overlapping mutable
    /// `f32` output slices simultaneously.
    ///
    /// Useful for ops that split or produce two separate outputs from a single
    /// input (e.g. `split_inner_dim`).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Any byte range exceeds the arena bounds
    /// - The two output ranges overlap each other
    /// - Either output range overlaps the input range
    /// - Any offset is not 4-byte aligned
    #[must_use]
    pub fn one_slice_in_two_out(
        &mut self,
        in_offset: usize,
        in_len: usize,
        out1_offset: usize,
        out1_len: usize,
        out2_offset: usize,
        out2_len: usize,
    ) -> (&[f32], &mut [f32], &mut [f32]) {
        let in_bytes = in_len * mem::size_of::<f32>();
        let out1_bytes = out1_len * mem::size_of::<f32>();
        let out2_bytes = out2_len * mem::size_of::<f32>();
        let in_end = in_offset + in_bytes;
        let out1_end = out1_offset + out1_bytes;
        let out2_end = out2_offset + out2_bytes;
        let arena_len = self.data.len();

        assert!(
            in_end <= arena_len,
            "in [{in_offset}..{in_end}) exceeds arena size {arena_len}"
        );
        assert!(
            out1_end <= arena_len,
            "out1 [{out1_offset}..{out1_end}) exceeds arena size {arena_len}"
        );
        assert!(
            out2_end <= arena_len,
            "out2 [{out2_offset}..{out2_end}) exceeds arena size {arena_len}"
        );
        // Outputs must not overlap each other.
        assert!(
            out1_end <= out2_offset || out2_end <= out1_offset,
            "out1 [{out1_offset}..{out1_end}) and out2 [{out2_offset}..{out2_end}) overlap"
        );
        // Outputs must not overlap the input.
        assert!(
            out1_end <= in_offset || in_end <= out1_offset,
            "out1 [{out1_offset}..{out1_end}) overlaps in [{in_offset}..{in_end})"
        );
        assert!(
            out2_end <= in_offset || in_end <= out2_offset,
            "out2 [{out2_offset}..{out2_end}) overlaps in [{in_offset}..{in_end})"
        );

        // SAFETY: `in` is cast to `*const` so it carries no exclusive-access
        // guarantee. `out1` and `out2` are separate `*mut` regions that we
        // verified are non-overlapping and non-aliasing with `in`.
        // All three ranges are within bounds.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let in_raw = std::slice::from_raw_parts(ptr.add(in_offset).cast_const(), in_bytes);
            let out1_raw = std::slice::from_raw_parts_mut(ptr.add(out1_offset), out1_bytes);
            let out2_raw = std::slice::from_raw_parts_mut(ptr.add(out2_offset), out2_bytes);
            (
                bytemuck::cast_slice(in_raw),
                bytemuck::cast_slice_mut(out1_raw),
                bytemuck::cast_slice_mut(out2_raw),
            )
        }
    }

    /// Borrow two immutable `f32` input slices and two non-overlapping mutable
    /// `f32` output slices simultaneously.
    ///
    /// Useful for fused ops that consume two inputs and produce two outputs
    /// (e.g. `add_rms_norm`).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Any byte range exceeds the arena bounds
    /// - The two output ranges overlap each other
    /// - Either output range overlaps either input range
    /// - Any offset is not 4-byte aligned
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn two_slices_in_two_out(
        &mut self,
        in1_offset: usize,
        in1_len: usize,
        in2_offset: usize,
        in2_len: usize,
        out1_offset: usize,
        out1_len: usize,
        out2_offset: usize,
        out2_len: usize,
    ) -> (&[f32], &[f32], &mut [f32], &mut [f32]) {
        let in1_bytes = in1_len * mem::size_of::<f32>();
        let in2_bytes = in2_len * mem::size_of::<f32>();
        let out1_bytes = out1_len * mem::size_of::<f32>();
        let out2_bytes = out2_len * mem::size_of::<f32>();
        let in1_end = in1_offset + in1_bytes;
        let in2_end = in2_offset + in2_bytes;
        let out1_end = out1_offset + out1_bytes;
        let out2_end = out2_offset + out2_bytes;
        let arena_len = self.data.len();

        assert!(
            in1_end <= arena_len,
            "in1 [{in1_offset}..{in1_end}) exceeds arena size {arena_len}"
        );
        assert!(
            in2_end <= arena_len,
            "in2 [{in2_offset}..{in2_end}) exceeds arena size {arena_len}"
        );
        assert!(
            out1_end <= arena_len,
            "out1 [{out1_offset}..{out1_end}) exceeds arena size {arena_len}"
        );
        assert!(
            out2_end <= arena_len,
            "out2 [{out2_offset}..{out2_end}) exceeds arena size {arena_len}"
        );
        // Outputs must not overlap each other.
        assert!(
            out1_end <= out2_offset || out2_end <= out1_offset,
            "out1 [{out1_offset}..{out1_end}) and out2 [{out2_offset}..{out2_end}) overlap"
        );
        // Each output must not overlap either input.
        assert!(
            out1_end <= in1_offset || in1_end <= out1_offset,
            "out1 [{out1_offset}..{out1_end}) overlaps in1 [{in1_offset}..{in1_end})"
        );
        assert!(
            out1_end <= in2_offset || in2_end <= out1_offset,
            "out1 [{out1_offset}..{out1_end}) overlaps in2 [{in2_offset}..{in2_end})"
        );
        assert!(
            out2_end <= in1_offset || in1_end <= out2_offset,
            "out2 [{out2_offset}..{out2_end}) overlaps in1 [{in1_offset}..{in1_end})"
        );
        assert!(
            out2_end <= in2_offset || in2_end <= out2_offset,
            "out2 [{out2_offset}..{out2_end}) overlaps in2 [{in2_offset}..{in2_end})"
        );

        // SAFETY: `in1` and `in2` are cast to `*const` so they carry no
        // exclusive-access guarantee. `out1` and `out2` are separate `*mut`
        // regions that we verified are non-overlapping with each other and with
        // both inputs. All four ranges are within bounds.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let in1_raw = std::slice::from_raw_parts(ptr.add(in1_offset).cast_const(), in1_bytes);
            let in2_raw = std::slice::from_raw_parts(ptr.add(in2_offset).cast_const(), in2_bytes);
            let out1_raw = std::slice::from_raw_parts_mut(ptr.add(out1_offset), out1_bytes);
            let out2_raw = std::slice::from_raw_parts_mut(ptr.add(out2_offset), out2_bytes);
            (
                bytemuck::cast_slice(in1_raw),
                bytemuck::cast_slice(in2_raw),
                bytemuck::cast_slice_mut(out1_raw),
                bytemuck::cast_slice_mut(out2_raw),
            )
        }
    }

    /// Borrow one immutable `u32` input slice and one mutable `f32` output
    /// slice simultaneously.
    ///
    /// Useful for ops like `embedding_gather` that read token IDs (`u32`) and
    /// write float activations.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Either byte range exceeds the arena bounds
    /// - The output range overlaps the input range
    /// - Either offset is not 4-byte aligned
    #[must_use]
    pub fn u32_slice_in_f32_out(
        &mut self,
        in_offset: usize,
        in_len: usize,
        out_offset: usize,
        out_len: usize,
    ) -> (&[u32], &mut [f32]) {
        let in_bytes = in_len * mem::size_of::<u32>();
        let out_bytes = out_len * mem::size_of::<f32>();
        let in_end = in_offset + in_bytes;
        let out_end = out_offset + out_bytes;
        let arena_len = self.data.len();

        assert!(
            in_end <= arena_len,
            "in [{in_offset}..{in_end}) exceeds arena size {arena_len}"
        );
        assert!(
            out_end <= arena_len,
            "out [{out_offset}..{out_end}) exceeds arena size {arena_len}"
        );
        // The mutable output must not overlap the immutable input.
        assert!(
            out_end <= in_offset || in_end <= out_offset,
            "out [{out_offset}..{out_end}) overlaps in [{in_offset}..{in_end})"
        );

        // SAFETY: `in` is cast to `*const` so it carries no exclusive-access
        // guarantee. `out` is the sole `*mut` region, and we verified it does
        // not overlap with the input. Both ranges are within bounds.
        // `u32` and `f32` have the same size (4 bytes) and alignment, and the
        // planner guarantees 4-byte aligned offsets.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let in_raw = std::slice::from_raw_parts(ptr.add(in_offset).cast_const(), in_bytes);
            let out_raw = std::slice::from_raw_parts_mut(ptr.add(out_offset), out_bytes);
            (
                bytemuck::cast_slice(in_raw),
                bytemuck::cast_slice_mut(out_raw),
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

    // --- two_slices_in_one_out ---

    #[test]
    fn two_in_one_out_correct_values() {
        // Layout: [in1: 4 f32][in2: 4 f32][out: 4 f32] = 48 bytes
        let mut arena = Arena::new(48);
        // Write in1 and in2 via the simple helpers first.
        let s = arena.f32_slice_mut(0, 4);
        s.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let s = arena.f32_slice_mut(16, 4);
        s.copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

        let (in1, in2, out) = arena.two_slices_in_one_out(0, 4, 16, 4, 32, 4);
        assert_eq!(in1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(in2, &[10.0, 20.0, 30.0, 40.0]);
        for i in 0..4 {
            out[i] = in1[i] + in2[i];
        }
        drop((in1, in2, out));

        let result = arena.f32_slice(32, 4);
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[should_panic(expected = "overlaps in1")]
    fn two_in_one_out_out_overlaps_in1_panics() {
        let mut arena = Arena::new(48);
        // out at offset 0 overlaps in1 at offset 0.
        let _ = arena.two_slices_in_one_out(0, 4, 16, 4, 0, 4);
    }

    #[test]
    #[should_panic(expected = "overlaps in2")]
    fn two_in_one_out_out_overlaps_in2_panics() {
        let mut arena = Arena::new(48);
        // out at offset 16 overlaps in2 at offset 16.
        let _ = arena.two_slices_in_one_out(0, 4, 16, 4, 16, 4);
    }

    // --- one_slice_in_two_out ---

    #[test]
    fn one_in_two_out_correct_values() {
        // Layout: [in: 4 f32][out1: 2 f32][out2: 2 f32] = 32 bytes
        let mut arena = Arena::new(32);
        let s = arena.f32_slice_mut(0, 4);
        s.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let (inp, out1, out2) = arena.one_slice_in_two_out(0, 4, 16, 2, 24, 2);
        out1[0] = inp[0];
        out1[1] = inp[1];
        out2[0] = inp[2];
        out2[1] = inp[3];
        drop((inp, out1, out2));

        let first = arena.f32_slice(16, 2);
        assert_eq!(first, &[1.0, 2.0]);
        let second = arena.f32_slice(24, 2);
        assert_eq!(second, &[3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn one_in_two_out_outputs_overlap_panics() {
        let mut arena = Arena::new(32);
        // out1 [16..24) and out2 [20..28) overlap.
        let _ = arena.one_slice_in_two_out(0, 4, 16, 2, 20, 2);
    }

    #[test]
    #[should_panic(expected = "overlaps in")]
    fn one_in_two_out_out1_overlaps_in_panics() {
        let mut arena = Arena::new(32);
        // out1 starts at 0, same as in.
        let _ = arena.one_slice_in_two_out(0, 4, 0, 2, 24, 2);
    }

    // --- two_slices_in_two_out ---

    #[test]
    fn two_in_two_out_correct_values() {
        // Layout: [in1: 4 f32][in2: 4 f32][out1: 4 f32][out2: 4 f32] = 64 bytes
        let mut arena = Arena::new(64);
        let s = arena.f32_slice_mut(0, 4);
        s.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let s = arena.f32_slice_mut(16, 4);
        s.copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

        let (in1, in2, out1, out2) = arena.two_slices_in_two_out(0, 4, 16, 4, 32, 4, 48, 4);
        for i in 0..4 {
            out1[i] = in1[i] + in2[i];
            out2[i] = in1[i] * in2[i];
        }
        drop((in1, in2, out1, out2));

        let sums = arena.f32_slice(32, 4);
        assert_eq!(sums, &[6.0, 8.0, 10.0, 12.0]);
        let products = arena.f32_slice(48, 4);
        assert_eq!(products, &[5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn two_in_two_out_outputs_overlap_panics() {
        let mut arena = Arena::new(64);
        // out1 [32..48) and out2 [40..56) overlap.
        let _ = arena.two_slices_in_two_out(0, 4, 16, 4, 32, 4, 40, 4);
    }

    #[test]
    #[should_panic(expected = "overlaps in1")]
    fn two_in_two_out_out1_overlaps_in1_panics() {
        let mut arena = Arena::new(64);
        // out1 at offset 0 overlaps in1 at offset 0.
        let _ = arena.two_slices_in_two_out(0, 4, 16, 4, 0, 4, 48, 4);
    }

    // --- u32_slice_in_f32_out ---

    #[test]
    fn u32_in_f32_out_correct_values() {
        // Layout: [in: 4 u32][out: 4 f32] = 32 bytes
        let mut arena = Arena::new(32);
        let s = arena.u32_slice_mut(0, 4);
        s.copy_from_slice(&[0u32, 1, 2, 3]);

        let (ids, out) = arena.u32_slice_in_f32_out(0, 4, 16, 4);
        for i in 0..4 {
            out[i] = ids[i] as f32 * 10.0;
        }
        drop((ids, out));

        let result = arena.f32_slice(16, 4);
        assert_eq!(result, &[0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    #[should_panic(expected = "overlaps in")]
    fn u32_in_f32_out_overlap_panics() {
        let mut arena = Arena::new(32);
        // out starts at 0, same as in — overlap.
        let _ = arena.u32_slice_in_f32_out(0, 4, 0, 4);
    }

    #[test]
    #[should_panic(expected = "exceeds arena size")]
    fn u32_in_f32_out_oob_panics() {
        let mut arena = Arena::new(16);
        // in is 4 u32 = 16 bytes, out is 4 f32 = 16 bytes, total 32 > 16.
        let _ = arena.u32_slice_in_f32_out(0, 4, 16, 4);
    }
}
