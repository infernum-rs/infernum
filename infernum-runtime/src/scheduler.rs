//! Backend-generic request scheduling for inflight (continuous) batching.
//!
//! [`Scheduler`] tracks per-request state and decides which requests to
//! process at each iteration. It implements FCFS admission with block-level
//! memory accounting.
//!
//! This module is the backend-agnostic replacement for [`Scheduler`](super::Scheduler).
//! No CUDA-specific imports — all types use `infernum` core.

#![allow(clippy::module_name_repetitions, clippy::missing_panics_doc)]

use std::collections::VecDeque;
use std::sync::mpsc;

use infernum::{BlockAllocator, BlockTable, GenerateOptions};

// ---------------------------------------------------------------------------
// Public types (moved from engine, no CUDA dependency)
// ---------------------------------------------------------------------------

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Model produced the end-of-sequence token.
    Stop,
    /// Reached the maximum number of tokens.
    Length,
    /// The receiver was dropped (client disconnect).
    Cancelled,
}

/// An event produced by the engine during generation.
///
/// Sent through a single channel so ordering is guaranteed:
/// zero or more `Token`s, then exactly one terminal event
/// (`Finished` or `Error`).
pub enum GenerationEvent {
    /// A newly generated token.
    Token(u32),
    /// An error occurred during generation.
    Error(infernum::Error),
    /// Generation completed with the given reason.
    Finished(FinishReason),
}

/// Trait for sending generation events from the engine to the caller.
///
/// Abstracted so callers can provide either a sync or async sender.
/// Return `false` to signal that the receiver has been dropped and
/// generation should stop.
pub trait TokenSender: Send {
    /// Send a generation event to the receiver.
    ///
    /// Returns `false` if the receiver has been dropped, signalling the
    /// engine to abort generation early.
    fn send(&self, event: GenerationEvent) -> bool;
}

impl TokenSender for mpsc::Sender<GenerationEvent> {
    fn send(&self, event: GenerationEvent) -> bool {
        mpsc::Sender::send(self, event).is_ok()
    }
}

impl TokenSender for Box<dyn TokenSender> {
    fn send(&self, event: GenerationEvent) -> bool {
        (**self).send(event)
    }
}

/// Configuration for the batched engine / scheduler.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of sequences in the running batch at once.
    pub max_batch_size: usize,
    /// Maximum number of prompt tokens to process in a single prefill chunk.
    /// Larger values improve prefill throughput but increase latency for
    /// decode-phase requests waiting for their turn.
    pub max_prefill_tokens: usize,
    /// Number of tokens per KV cache block.
    pub block_size: usize,
    /// Total number of KV cache blocks in the pool.
    pub num_blocks: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_prefill_tokens: 512,
            block_size: 16,
            num_blocks: 1024,
        }
    }
}

/// Phase of a sequence's lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencePhase {
    /// Waiting in the queue, not yet started.
    Waiting,
    /// Prefilling (processing the prompt).
    Prefill,
    /// Decoding (generating tokens one at a time).
    Decode,
    /// Finished (EOS, max tokens, or cancelled).
    Finished(FinishReason),
}

/// Per-request state tracked by the scheduler.
pub struct SequenceState {
    /// Unique request ID.
    pub id: u64,
    /// Input token IDs (full prompt).
    pub prompt_ids: Vec<u32>,
    /// Generated token IDs so far (decode phase).
    pub generated_ids: Vec<u32>,
    /// Block table for this sequence's KV cache.
    pub block_table: BlockTable,
    /// Current phase.
    pub phase: SequencePhase,
    /// Generation options (max tokens, sampling, EOS).
    pub options: GenerateOptions,
    /// Channel for sending tokens back to the caller.
    pub token_tx: Box<dyn TokenSender>,
    /// Number of prompt tokens processed so far (for chunked prefill).
    pub prefill_progress: usize,
}

impl SequenceState {
    /// Create a new sequence in the `Waiting` phase.
    #[must_use]
    pub fn new(
        id: u64,
        prompt_ids: Vec<u32>,
        options: GenerateOptions,
        token_tx: Box<dyn TokenSender>,
        block_size: usize,
    ) -> Self {
        Self {
            id,
            prompt_ids,
            generated_ids: Vec::new(),
            block_table: BlockTable::new(block_size),
            phase: SequencePhase::Waiting,
            options,
            token_tx,
            prefill_progress: 0,
        }
    }

    /// Total number of tokens this sequence occupies in the KV cache
    /// (prompt tokens processed + generated tokens).
    #[must_use]
    pub fn kv_len(&self) -> usize {
        self.block_table.seq_len()
    }

    /// Current position for the next token (= number of tokens in the cache).
    #[must_use]
    pub fn current_position(&self) -> usize {
        self.kv_len()
    }
}

/// A prefill task for one sequence in a single iteration.
pub struct PrefillTask {
    /// Index into the scheduler's `running` list.
    pub running_idx: usize,
    /// Token IDs to process this iteration (a slice of the full prompt).
    pub token_ids: Vec<u32>,
}

/// A decode task for one sequence in a single iteration.
pub struct DecodeTask {
    /// Index into the scheduler's `running` list.
    pub running_idx: usize,
}

/// Output of one scheduling step.
pub struct SchedulerOutput {
    /// Sequences to prefill this iteration (one chunk each).
    pub prefill: Vec<PrefillTask>,
    /// Sequences to decode this iteration (one token each).
    pub decode: Vec<DecodeTask>,
    /// Indices of sequences in `running` that finished this iteration
    /// (collected in reverse-sorted order for safe removal).
    pub finished_indices: Vec<usize>,
}

/// Backend-generic iteration-level scheduler for inflight batching.
///
/// Manages a waiting queue and a running batch. At each step, it:
/// 1. Identifies finished sequences
/// 2. Continues all decode-phase sequences
/// 3. Admits new requests from the waiting queue (FCFS) if blocks are available
pub struct Scheduler {
    /// Requests waiting to be admitted.
    waiting: VecDeque<SequenceState>,
    /// Currently running requests (prefill or decode phase).
    running: Vec<SequenceState>,
    /// Maximum batch size (concurrent sequences).
    max_batch_size: usize,
    /// Maximum prompt tokens per prefill chunk.
    max_prefill_tokens: usize,
    /// Tokens per block (for computing block requirements).
    block_size: usize,
    /// Next unique request ID.
    next_id: u64,
}

impl Scheduler {
    /// Create a new scheduler.
    #[must_use]
    pub fn new(config: &BatchConfig) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            max_batch_size: config.max_batch_size,
            max_prefill_tokens: config.max_prefill_tokens,
            block_size: config.block_size,
            next_id: 0,
        }
    }

    /// Submit a new request. Returns the assigned request ID.
    #[must_use]
    pub fn add_request(
        &mut self,
        prompt_ids: Vec<u32>,
        options: GenerateOptions,
        token_tx: Box<dyn TokenSender>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let state = SequenceState::new(id, prompt_ids, options, token_tx, self.block_size);
        self.waiting.push_back(state);
        id
    }

    /// Add a pre-built `SequenceState` directly (for testing or migration).
    pub fn add_sequence(&mut self, state: SequenceState) {
        self.waiting.push_back(state);
    }

    /// Whether there is nothing to do (no waiting or running requests).
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Number of sequences currently in the running batch.
    #[must_use]
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Number of sequences in the waiting queue.
    #[must_use]
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Mutable access to running sequences (for the engine to update state).
    pub fn running_mut(&mut self) -> &mut Vec<SequenceState> {
        &mut self.running
    }

    /// Immutable access to running sequences.
    #[must_use]
    pub fn running(&self) -> &[SequenceState] {
        &self.running
    }

    /// Run one scheduling step.
    ///
    /// The `allocator` is used to check block availability and allocate
    /// blocks for newly admitted requests.
    ///
    /// Returns a [`SchedulerOutput`] describing what the engine should do
    /// this iteration.
    pub fn step(&mut self, allocator: &mut BlockAllocator) -> SchedulerOutput {
        let mut output = SchedulerOutput {
            prefill: Vec::new(),
            decode: Vec::new(),
            finished_indices: Vec::new(),
        };

        // 1. Identify finished sequences (reverse order for safe removal)
        for i in (0..self.running.len()).rev() {
            if matches!(self.running[i].phase, SequencePhase::Finished(_)) {
                output.finished_indices.push(i);
            }
        }

        // 2. Remove finished sequences and free their blocks
        // finished_indices is already in reverse order
        let mut finished = Vec::new();
        for &idx in &output.finished_indices {
            let seq = self.running.remove(idx);
            allocator.free_all(seq.block_table.blocks());
            finished.push(seq);
        }
        // Drop finished sequences (closes token channels)
        drop(finished);

        // Rebuild output indices after removal (they're no longer valid)
        output.finished_indices.clear();

        // 3. Schedule decode tasks for all running decode-phase sequences
        for (i, seq) in self.running.iter().enumerate() {
            if seq.phase == SequencePhase::Decode {
                output.decode.push(DecodeTask { running_idx: i });
            }
        }

        // 4. Admit new requests from the waiting queue (FCFS)
        while self.running.len() < self.max_batch_size {
            let Some(seq) = self.waiting.front() else {
                break;
            };

            // How many blocks does the prompt need?
            let prompt_len = seq.prompt_ids.len();
            let blocks_needed = prompt_len.div_ceil(self.block_size);

            if !allocator.can_allocate(blocks_needed) {
                // Not enough memory — stop admitting (FCFS, don't skip)
                break;
            }

            // Admit: allocate blocks and move to running
            let mut seq = self.waiting.pop_front().unwrap();
            for _ in 0..blocks_needed {
                let block = allocator.allocate().expect("checked can_allocate");
                seq.block_table.append_block(block);
            }
            seq.phase = SequencePhase::Prefill;

            // Build the prefill task (possibly chunked)
            let remaining = prompt_len - seq.prefill_progress;
            let chunk_size = remaining.min(self.max_prefill_tokens);
            let chunk_start = seq.prefill_progress;
            let token_ids = seq.prompt_ids[chunk_start..chunk_start + chunk_size].to_vec();

            let running_idx = self.running.len();
            self.running.push(seq);

            output.prefill.push(PrefillTask {
                running_idx,
                token_ids,
            });
        }

        output
    }

    /// Continue a prefill that was chunked. Returns `Some(PrefillTask)` if
    /// more chunks remain, or `None` if prefill is complete (sequence
    /// transitions to `Decode`).
    ///
    /// Called by the engine after processing a prefill chunk.
    pub fn continue_prefill(&mut self, running_idx: usize) -> Option<PrefillTask> {
        let seq = &mut self.running[running_idx];
        let remaining = seq.prompt_ids.len() - seq.prefill_progress;

        if remaining == 0 {
            // Prefill complete → transition to decode
            seq.phase = SequencePhase::Decode;
            return None;
        }

        let chunk_size = remaining.min(self.max_prefill_tokens);
        let chunk_start = seq.prefill_progress;
        let token_ids = seq.prompt_ids[chunk_start..chunk_start + chunk_size].to_vec();

        Some(PrefillTask {
            running_idx,
            token_ids,
        })
    }

    /// Mark a running sequence as finished.
    pub fn finish_sequence(&mut self, running_idx: usize, reason: FinishReason) {
        self.running[running_idx].phase = SequencePhase::Finished(reason);
    }

    /// Allocate a new block for a running sequence that needs more space.
    ///
    /// Returns `true` if allocation succeeded, `false` if the pool is
    /// exhausted.
    pub fn allocate_block_for(
        &mut self,
        running_idx: usize,
        allocator: &mut BlockAllocator,
    ) -> bool {
        if let Some(block) = allocator.allocate() {
            self.running[running_idx].block_table.append_block(block);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::BlockConfig;

    fn default_options() -> GenerateOptions {
        GenerateOptions {
            max_new_tokens: 10,
            eos_token_id: Some(2),
            ..Default::default()
        }
    }

    fn make_scheduler(max_batch: usize, block_size: usize, num_blocks: usize) -> Scheduler {
        Scheduler::new(&BatchConfig {
            max_batch_size: max_batch,
            max_prefill_tokens: 512,
            block_size,
            num_blocks,
        })
    }

    fn make_allocator(block_size: usize, num_blocks: usize) -> BlockAllocator {
        BlockAllocator::new(&BlockConfig {
            block_size,
            num_blocks,
        })
    }

    #[test]
    fn idle_when_empty() {
        let sched = make_scheduler(4, 16, 64);
        assert!(sched.is_idle());
        assert_eq!(sched.num_running(), 0);
        assert_eq!(sched.num_waiting(), 0);
    }

    #[test]
    fn add_request_goes_to_waiting() {
        let mut sched = make_scheduler(4, 16, 64);
        let (tx, _rx) = mpsc::channel();
        let _ = sched.add_request(vec![1, 2, 3], default_options(), Box::new(tx));

        assert!(!sched.is_idle());
        assert_eq!(sched.num_waiting(), 1);
        assert_eq!(sched.num_running(), 0);
    }

    #[test]
    fn step_admits_request_to_prefill() {
        let mut sched = make_scheduler(4, 4, 64);
        let mut alloc = make_allocator(4, 64);
        let (tx, _rx) = mpsc::channel();

        // Prompt of 5 tokens → needs ceil(5/4) = 2 blocks
        let _ = sched.add_request(vec![1, 2, 3, 4, 5], default_options(), Box::new(tx));

        let output = sched.step(&mut alloc);

        assert_eq!(output.prefill.len(), 1);
        assert_eq!(output.decode.len(), 0);
        assert_eq!(sched.num_running(), 1);
        assert_eq!(sched.num_waiting(), 0);
        assert_eq!(alloc.num_free(), 62); // 64 - 2 blocks

        // Check prefill task has the right tokens
        assert_eq!(output.prefill[0].token_ids, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn decode_phase_sequences_continue() {
        let mut sched = make_scheduler(4, 4, 64);
        let mut alloc = make_allocator(4, 64);
        let (tx, _rx) = mpsc::channel();

        let _ = sched.add_request(vec![1, 2, 3], default_options(), Box::new(tx));

        // Step 1: admit to prefill
        let _ = sched.step(&mut alloc);

        // Simulate prefill complete → transition to decode
        sched.running_mut()[0].prefill_progress = 3;
        sched.running_mut()[0].phase = SequencePhase::Decode;

        // Step 2: should schedule as decode
        let output = sched.step(&mut alloc);
        assert_eq!(output.prefill.len(), 0);
        assert_eq!(output.decode.len(), 1);
    }

    #[test]
    fn batch_size_limit() {
        let mut sched = make_scheduler(2, 4, 64);
        let mut alloc = make_allocator(4, 64);

        for _ in 0..4 {
            let (tx, _rx) = mpsc::channel();
            let _ = sched.add_request(vec![1, 2], default_options(), Box::new(tx));
        }

        let output = sched.step(&mut alloc);
        // Only 2 admitted (max_batch_size = 2)
        assert_eq!(output.prefill.len(), 2);
        assert_eq!(sched.num_running(), 2);
        assert_eq!(sched.num_waiting(), 2);
    }

    #[test]
    fn block_exhaustion_stops_admission() {
        let mut sched = make_scheduler(8, 4, 3);
        let mut alloc = make_allocator(4, 3);

        // Request 1: 4 tokens → 1 block
        let (tx1, _rx1) = mpsc::channel();
        let _ = sched.add_request(vec![1, 2, 3, 4], default_options(), Box::new(tx1));

        // Request 2: 8 tokens → 2 blocks
        let (tx2, _rx2) = mpsc::channel();
        let _ = sched.add_request(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            default_options(),
            Box::new(tx2),
        );

        // Request 3: 4 tokens → 1 block (only 0 left after first two)
        let (tx3, _rx3) = mpsc::channel();
        let _ = sched.add_request(vec![1, 2, 3, 4], default_options(), Box::new(tx3));

        let output = sched.step(&mut alloc);
        // Only first 2 admitted (1 + 2 = 3 blocks, pool exhausted)
        assert_eq!(output.prefill.len(), 2);
        assert_eq!(sched.num_running(), 2);
        assert_eq!(sched.num_waiting(), 1);
        assert_eq!(alloc.num_free(), 0);
    }

    #[test]
    fn finished_sequences_are_cleaned_up() {
        let mut sched = make_scheduler(4, 4, 64);
        let mut alloc = make_allocator(4, 64);
        let (tx, rx) = mpsc::channel::<GenerationEvent>();

        let _ = sched.add_request(vec![1, 2, 3], default_options(), Box::new(tx));
        let _ = sched.step(&mut alloc);

        let blocks_before = alloc.num_free();

        // Mark as finished
        sched.finish_sequence(0, FinishReason::Stop);

        // Next step should clean up
        let _ = sched.step(&mut alloc);

        assert_eq!(sched.num_running(), 0);
        assert!(alloc.num_free() > blocks_before);

        // Channel should be closed (sender dropped)
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn fcfs_ordering() {
        let mut sched = make_scheduler(1, 4, 64);
        let mut alloc = make_allocator(4, 64);

        let (tx1, _rx1) = mpsc::channel();
        let (tx2, _rx2) = mpsc::channel();
        let _ = sched.add_request(vec![10, 20], default_options(), Box::new(tx1));
        let _ = sched.add_request(vec![30, 40], default_options(), Box::new(tx2));

        // Only first request admitted (batch size 1)
        let output = sched.step(&mut alloc);
        assert_eq!(output.prefill.len(), 1);
        assert_eq!(output.prefill[0].token_ids, vec![10, 20]);

        // Finish first, admit second
        sched.running_mut()[0].prefill_progress = 2;
        sched.finish_sequence(0, FinishReason::Stop);

        let output = sched.step(&mut alloc);
        assert_eq!(output.prefill.len(), 1);
        assert_eq!(output.prefill[0].token_ids, vec![30, 40]);
    }

    #[test]
    fn chunked_prefill() {
        let mut sched = Scheduler::new(&BatchConfig {
            max_batch_size: 4,
            max_prefill_tokens: 3, // Very small for testing
            block_size: 4,
            num_blocks: 64,
        });
        let mut alloc = make_allocator(4, 64);
        let (tx, _rx) = mpsc::channel();

        // 7-token prompt, chunks of 3: [0..3], [3..6], [6..7]
        let _ = sched.add_request(vec![1, 2, 3, 4, 5, 6, 7], default_options(), Box::new(tx));

        let output = sched.step(&mut alloc);
        assert_eq!(output.prefill.len(), 1);
        assert_eq!(output.prefill[0].token_ids, vec![1, 2, 3]); // first chunk

        // Simulate processing the chunk
        sched.running_mut()[0].prefill_progress = 3;

        // Continue prefill
        let next = sched.continue_prefill(0);
        assert!(next.is_some());
        assert_eq!(next.unwrap().token_ids, vec![4, 5, 6]);

        sched.running_mut()[0].prefill_progress = 6;

        let next = sched.continue_prefill(0);
        assert!(next.is_some());
        assert_eq!(next.unwrap().token_ids, vec![7]);

        sched.running_mut()[0].prefill_progress = 7;

        // No more chunks → transitions to decode
        let next = sched.continue_prefill(0);
        assert!(next.is_none());
        assert_eq!(sched.running()[0].phase, SequencePhase::Decode);
    }

    #[test]
    fn allocate_block_for_sequence() {
        let mut sched = make_scheduler(4, 4, 4);
        let mut alloc = make_allocator(4, 4);
        let (tx, _rx) = mpsc::channel();

        let _ = sched.add_request(vec![1], default_options(), Box::new(tx));
        let _ = sched.step(&mut alloc); // uses 1 block

        assert_eq!(alloc.num_free(), 3);

        // Allocate another block for the sequence
        assert!(sched.allocate_block_for(0, &mut alloc));
        assert_eq!(alloc.num_free(), 2);
        assert_eq!(sched.running()[0].block_table.num_blocks(), 2);
    }

    #[test]
    fn interleaved_prefill_and_decode() {
        let mut sched = make_scheduler(4, 4, 64);
        let mut alloc = make_allocator(4, 64);

        // Request 1: already decoding
        let (tx1, _rx1) = mpsc::channel();
        let _ = sched.add_request(vec![1, 2], default_options(), Box::new(tx1));
        let _ = sched.step(&mut alloc);
        sched.running_mut()[0].prefill_progress = 2;
        sched.running_mut()[0].phase = SequencePhase::Decode;

        // Request 2: new arrival
        let (tx2, _rx2) = mpsc::channel();
        let _ = sched.add_request(vec![10, 20, 30], default_options(), Box::new(tx2));

        let output = sched.step(&mut alloc);
        // Should have 1 decode (request 1) and 1 prefill (request 2)
        assert_eq!(output.decode.len(), 1);
        assert_eq!(output.prefill.len(), 1);
        assert_eq!(output.prefill[0].token_ids, vec![10, 20, 30]);
    }
}
