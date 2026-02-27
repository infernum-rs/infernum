//! Pre-allocated GPU buffers for CUDA graph-captured batched decode.
//!
//! All buffers are allocated at engine start with `max_batch_size` capacity.
//! Contents are updated via `htod_copy_into` between graph replays.
//!
//! Currently unused: will be wired into `DecodeBufferOps` when CUDA graph
//! capture is implemented.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc
)]

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use infernum::Result;

/// Pre-allocated GPU buffers for CUDA graph-captured batched decode.
///
/// During graph replay, kernel arguments (token IDs, positions, block tables,
/// `seq_lens`) must live at fixed GPU addresses. This struct holds those
/// buffers, allocated once at engine start, and provides an `update` method
/// to write new values before each replay.
///
/// Padding sequences (beyond `actual_batch_size`) use:
/// - `token_id` = 0
/// - `position` = 0
/// - `block_table` entries = `dummy_block` index
/// - `seq_len` = 1
pub struct BatchedGraphInputs {
    /// Token IDs: shape `(max_batch_size,)`
    token_ids: CudaSlice<u32>,
    /// Positions: shape `(max_batch_size,)` — stored as `i32` to match CUDA kernels.
    positions: CudaSlice<i32>,
    /// Flattened block tables: shape `(max_batch_size * max_blocks_per_seq,)`
    block_tables: CudaSlice<i32>,
    /// Sequence lengths: shape `(max_batch_size,)`
    seq_lens: CudaSlice<i32>,
    /// Current actual batch size (host-side, for post-graph sampling).
    actual_batch_size: usize,
    max_batch_size: usize,
    max_blocks_per_seq: usize,
    /// Block index used for padding sequences.
    dummy_block: usize,
    device: Arc<CudaDevice>,
}

impl BatchedGraphInputs {
    /// Allocate all buffers on the given device.
    ///
    /// `dummy_block` is the physical block index that padding sequences'
    /// block tables point to — it must be pre-allocated and never freed.
    ///
    /// # Errors
    /// Returns an error if GPU allocation fails.
    pub fn new(
        device: &Arc<CudaDevice>,
        max_batch_size: usize,
        max_blocks_per_seq: usize,
        dummy_block: usize,
    ) -> Result<Self> {
        let token_ids = device.alloc_zeros::<u32>(max_batch_size)?;
        let positions = device.alloc_zeros::<i32>(max_batch_size)?;
        let block_tables = device.alloc_zeros::<i32>(max_batch_size * max_blocks_per_seq)?;
        let seq_lens = device.alloc_zeros::<i32>(max_batch_size)?;

        Ok(Self {
            token_ids,
            positions,
            block_tables,
            seq_lens,
            actual_batch_size: 0,
            max_batch_size,
            max_blocks_per_seq,
            dummy_block,
            device: Arc::clone(device),
        })
    }

    /// Update all buffers with new values, padding to `max_batch_size`.
    ///
    /// `token_ids`, `positions`, `block_tables_flat`, and `seq_lens` contain
    /// only the real sequences (`actual_batch_size` entries). This method
    /// pads them to `max_batch_size` and writes to the pre-allocated GPU
    /// buffers via `htod_copy_into`.
    ///
    /// `block_tables_flat` is already flattened to `(actual_batch_size, max_blocks_per_seq)`,
    /// padded with zeros for unused block slots within each row.
    ///
    /// # Errors
    /// Returns an error if the host-to-device copy fails.
    pub fn update(
        &mut self,
        token_ids: &[u32],
        positions: &[i32],
        block_tables_flat: &[i32],
        seq_lens: &[i32],
    ) -> Result<()> {
        let actual = token_ids.len();
        assert_eq!(positions.len(), actual);
        assert_eq!(seq_lens.len(), actual);
        assert_eq!(block_tables_flat.len(), actual * self.max_blocks_per_seq);
        assert!(
            actual <= self.max_batch_size,
            "actual batch size ({actual}) exceeds max ({})",
            self.max_batch_size
        );

        self.actual_batch_size = actual;
        let pad = self.max_batch_size - actual;

        // Token IDs: pad with 0
        let mut tids = token_ids.to_vec();
        tids.extend(std::iter::repeat_n(0u32, pad));
        self.device.htod_copy_into(tids, &mut self.token_ids)?;

        // Positions: pad with 0
        let mut pos = positions.to_vec();
        pos.extend(std::iter::repeat_n(0i32, pad));
        self.device.htod_copy_into(pos, &mut self.positions)?;

        // Block tables: pad rows with dummy_block index
        let dummy = self.dummy_block as i32;
        let mut bt = block_tables_flat.to_vec();
        for _ in 0..pad {
            let mut row = vec![dummy; self.max_blocks_per_seq];
            row[0] = dummy;
            bt.extend(row);
        }
        self.device.htod_copy_into(bt, &mut self.block_tables)?;

        // Seq lens: pad with 1 (minimum valid — attention over 1 KV entry)
        let mut sl = seq_lens.to_vec();
        sl.extend(std::iter::repeat_n(1i32, pad));
        self.device.htod_copy_into(sl, &mut self.seq_lens)?;

        Ok(())
    }

    /// GPU buffer holding token IDs.
    #[must_use]
    pub fn token_ids(&self) -> &CudaSlice<u32> {
        &self.token_ids
    }

    /// GPU buffer holding positions.
    #[must_use]
    pub fn positions(&self) -> &CudaSlice<i32> {
        &self.positions
    }

    /// GPU buffer holding flattened block tables.
    #[must_use]
    pub fn block_tables(&self) -> &CudaSlice<i32> {
        &self.block_tables
    }

    /// GPU buffer holding sequence lengths.
    #[must_use]
    pub fn seq_lens(&self) -> &CudaSlice<i32> {
        &self.seq_lens
    }

    /// Current actual batch size (set by last `update` call).
    #[must_use]
    pub fn actual_batch_size(&self) -> usize {
        self.actual_batch_size
    }

    /// Maximum batch size these buffers were allocated for.
    #[must_use]
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Maximum blocks per sequence.
    #[must_use]
    pub fn max_blocks_per_seq(&self) -> usize {
        self.max_blocks_per_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;
    use cudarc::driver::DeviceSlice;

    fn make_device() -> Arc<CudaDevice> {
        let ctx = CudaContext::new(0).expect("CUDA context");
        ctx.device().clone()
    }

    #[test]
    fn new_allocates_correct_sizes() {
        let device = make_device();
        let inputs = BatchedGraphInputs::new(&device, 4, 8, 0).unwrap();

        assert_eq!(inputs.max_batch_size(), 4);
        assert_eq!(inputs.max_blocks_per_seq(), 8);
        assert_eq!(inputs.actual_batch_size(), 0);
        assert_eq!(inputs.token_ids().len(), 4);
        assert_eq!(inputs.positions().len(), 4);
        assert_eq!(inputs.block_tables().len(), 32); // 4 * 8
        assert_eq!(inputs.seq_lens().len(), 4);
    }

    #[test]
    fn update_pads_correctly() {
        let device = make_device();
        let max_batch = 4;
        let max_blocks = 3;
        let dummy_block = 99;
        let mut inputs =
            BatchedGraphInputs::new(&device, max_batch, max_blocks, dummy_block).unwrap();

        // 2 real sequences
        let token_ids = [10u32, 20];
        let positions = [5i32, 12];
        let block_tables_flat = [
            1i32, 2, 0, // seq 0: blocks [1, 2, 0(unused)]
            3, 4, 5, // seq 1: blocks [3, 4, 5]
        ];
        let seq_lens = [6i32, 13];

        inputs
            .update(&token_ids, &positions, &block_tables_flat, &seq_lens)
            .unwrap();

        assert_eq!(inputs.actual_batch_size(), 2);

        // Verify GPU contents
        let tids = device.dtoh_sync_copy(inputs.token_ids()).unwrap();
        assert_eq!(tids, vec![10, 20, 0, 0]);

        let pos = device.dtoh_sync_copy(inputs.positions()).unwrap();
        assert_eq!(pos, vec![5, 12, 0, 0]);

        let sl = device.dtoh_sync_copy(inputs.seq_lens()).unwrap();
        assert_eq!(sl, vec![6, 13, 1, 1]);

        let bt = device.dtoh_sync_copy(inputs.block_tables()).unwrap();
        // seq 0: [1, 2, 0], seq 1: [3, 4, 5], padding: [99, 99, 99] x2
        assert_eq!(bt, vec![1, 2, 0, 3, 4, 5, 99, 99, 99, 99, 99, 99]);
    }

    #[test]
    fn update_full_batch_no_padding() {
        let device = make_device();
        let mut inputs = BatchedGraphInputs::new(&device, 2, 2, 0).unwrap();

        let token_ids = [7u32, 8];
        let positions = [3i32, 4];
        let block_tables_flat = [10i32, 11, 12, 13];
        let seq_lens = [4i32, 5];

        inputs
            .update(&token_ids, &positions, &block_tables_flat, &seq_lens)
            .unwrap();

        assert_eq!(inputs.actual_batch_size(), 2);

        let tids = device.dtoh_sync_copy(inputs.token_ids()).unwrap();
        assert_eq!(tids, vec![7, 8]);

        let sl = device.dtoh_sync_copy(inputs.seq_lens()).unwrap();
        assert_eq!(sl, vec![4, 5]);
    }

    #[test]
    #[should_panic(expected = "exceeds max")]
    fn update_exceeds_max_panics() {
        let device = make_device();
        let mut inputs = BatchedGraphInputs::new(&device, 2, 2, 0).unwrap();

        let token_ids = [1u32, 2, 3]; // 3 > max_batch_size=2
        let positions = [0i32, 0, 0];
        let block_tables_flat = [0i32, 0, 0, 0, 0, 0];
        let seq_lens = [1i32, 1, 1];

        inputs
            .update(&token_ids, &positions, &block_tables_flat, &seq_lens)
            .unwrap();
    }
}
