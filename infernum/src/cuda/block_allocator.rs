//! Block allocator for paged KV cache
//!
//! Manages a pool of fixed-size KV cache blocks. Each block holds
//! `block_size` tokens' worth of key and value data. Blocks are allocated
//! per-request as sequences grow and freed when requests finish.
//!
//! This is CPU-side bookkeeping only — the actual GPU memory lives in
//! [`PagedKvCache`](super::PagedKvCache).

#![allow(clippy::must_use_candidate)]

/// Configuration for the block allocator and paged KV cache.
#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    /// Number of tokens stored per block (e.g., 16).
    pub block_size: usize,
    /// Total number of blocks in the pool.
    pub num_blocks: usize,
}

/// Manages a pool of KV cache blocks.
///
/// Each block is identified by an index in `[0, num_blocks)`. The allocator
/// tracks which blocks are free using a simple stack (LIFO). Allocation is
/// O(1), freeing is O(1).
#[derive(Debug)]
pub struct BlockAllocator {
    free_blocks: Vec<usize>,
    num_blocks: usize,
    block_size: usize,
}

impl BlockAllocator {
    /// Create a new allocator with all blocks free.
    ///
    /// # Panics
    /// Panics if `block_size` or `num_blocks` is zero.
    pub fn new(config: &BlockConfig) -> Self {
        assert!(config.block_size > 0, "block_size must be > 0");
        assert!(config.num_blocks > 0, "num_blocks must be > 0");

        // Stack order: pop gives highest index first, but order doesn't matter
        let free_blocks: Vec<usize> = (0..config.num_blocks).collect();
        Self {
            free_blocks,
            num_blocks: config.num_blocks,
            block_size: config.block_size,
        }
    }

    /// Allocate a single block. Returns the block index, or `None` if
    /// the pool is exhausted.
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop()
    }

    /// Free a single block, returning it to the pool.
    ///
    /// # Panics
    /// Panics if `block_idx` is out of range or was already free.
    pub fn free(&mut self, block_idx: usize) {
        assert!(
            block_idx < self.num_blocks,
            "block index {block_idx} out of range (num_blocks = {})",
            self.num_blocks,
        );
        assert!(
            !self.free_blocks.contains(&block_idx),
            "double-free of block {block_idx}",
        );
        self.free_blocks.push(block_idx);
    }

    /// Free multiple blocks at once.
    ///
    /// # Panics
    /// Panics if any block index is out of range or was already free.
    pub fn free_all(&mut self, blocks: &[usize]) {
        for &block_idx in blocks {
            self.free(block_idx);
        }
    }

    /// Number of free blocks remaining.
    pub fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    /// Total number of blocks in the pool.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Number of tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Whether at least `n` blocks can be allocated.
    pub fn can_allocate(&self, n: usize) -> bool {
        self.free_blocks.len() >= n
    }

    /// Number of blocks needed to store `num_tokens` tokens.
    ///
    /// This is `ceil(num_tokens / block_size)`.
    pub fn blocks_needed(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }
}

/// Per-request block table mapping logical block indices to physical block
/// indices in the pool.
///
/// As a sequence grows, new physical blocks are appended to the table.
/// The attention kernel uses this table to look up where each token's
/// K/V data lives in the shared pool.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// `blocks[i]` = physical block index for logical block `i`.
    blocks: Vec<usize>,
    /// Current sequence length (number of tokens stored).
    seq_len: usize,
    /// Tokens per block (cached from allocator config).
    block_size: usize,
}

impl BlockTable {
    /// Create an empty block table.
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            seq_len: 0,
            block_size,
        }
    }

    /// Physical block indices in order.
    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of physical blocks allocated.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the current last block is full (the next append needs a new block).
    pub fn needs_new_block(&self) -> bool {
        self.blocks.is_empty() || self.seq_len.is_multiple_of(self.block_size)
    }

    /// Offset within the current (last) block where the next token should be written.
    pub fn current_offset(&self) -> usize {
        if self.blocks.is_empty() {
            0
        } else {
            self.seq_len % self.block_size
        }
    }

    /// Append a new physical block to the table.
    pub fn append_block(&mut self, block_idx: usize) {
        self.blocks.push(block_idx);
    }

    /// Record that `n` tokens have been appended to the KV cache.
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Reset the table for reuse (clears blocks and `seq_len`, does NOT free blocks).
    ///
    /// The caller must free the blocks via [`BlockAllocator::free_all`] before
    /// calling this.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.seq_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocator_all_free() {
        let alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 100,
        });
        assert_eq!(alloc.num_free(), 100);
        assert_eq!(alloc.num_blocks(), 100);
        assert_eq!(alloc.block_size(), 16);
    }

    #[test]
    fn allocate_and_free() {
        let mut alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 4,
        });

        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        assert_eq!(alloc.num_free(), 2);
        assert_ne!(b0, b1);

        alloc.free(b0);
        assert_eq!(alloc.num_free(), 3);

        alloc.free(b1);
        assert_eq!(alloc.num_free(), 4);
    }

    #[test]
    fn exhaustion_returns_none() {
        let mut alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 2,
        });

        assert!(alloc.allocate().is_some());
        assert!(alloc.allocate().is_some());
        assert!(alloc.allocate().is_none());
        assert_eq!(alloc.num_free(), 0);
    }

    #[test]
    fn can_allocate() {
        let mut alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 3,
        });

        assert!(alloc.can_allocate(3));
        assert!(!alloc.can_allocate(4));

        alloc.allocate();
        assert!(alloc.can_allocate(2));
        assert!(!alloc.can_allocate(3));
    }

    #[test]
    fn free_all() {
        let mut alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 4,
        });

        let blocks: Vec<usize> = (0..4).filter_map(|_| alloc.allocate()).collect();
        assert_eq!(alloc.num_free(), 0);

        alloc.free_all(&blocks);
        assert_eq!(alloc.num_free(), 4);
    }

    #[test]
    #[should_panic(expected = "double-free")]
    fn double_free_panics() {
        let mut alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 4,
        });

        let b = alloc.allocate().unwrap();
        alloc.free(b);
        alloc.free(b); // should panic
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn free_out_of_range_panics() {
        let mut alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 4,
        });

        alloc.free(10); // should panic
    }

    #[test]
    fn blocks_needed() {
        let alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 100,
        });

        assert_eq!(alloc.blocks_needed(0), 0);
        assert_eq!(alloc.blocks_needed(1), 1);
        assert_eq!(alloc.blocks_needed(16), 1);
        assert_eq!(alloc.blocks_needed(17), 2);
        assert_eq!(alloc.blocks_needed(32), 2);
        assert_eq!(alloc.blocks_needed(33), 3);
    }

    #[test]
    fn reuse_after_free() {
        let mut alloc = BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 2,
        });

        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        assert!(alloc.allocate().is_none());

        alloc.free(b0);
        let b2 = alloc.allocate().unwrap();
        assert_eq!(b2, b0); // LIFO reuse
        assert!(alloc.allocate().is_none());

        alloc.free(b1);
        alloc.free(b2);
        assert_eq!(alloc.num_free(), 2);
    }

    // ---- BlockTable tests ----

    #[test]
    fn empty_block_table() {
        let table = BlockTable::new(16);
        assert_eq!(table.seq_len(), 0);
        assert_eq!(table.num_blocks(), 0);
        assert!(table.blocks().is_empty());
        assert!(table.needs_new_block());
        assert_eq!(table.current_offset(), 0);
    }

    #[test]
    fn block_table_lifecycle() {
        let mut table = BlockTable::new(4); // 4 tokens per block

        // First token needs a new block
        assert!(table.needs_new_block());
        table.append_block(7); // physical block 7
        table.advance(1);
        assert_eq!(table.seq_len(), 1);
        assert_eq!(table.num_blocks(), 1);
        assert_eq!(table.current_offset(), 1);
        assert!(!table.needs_new_block());

        // Fill up the block (tokens 2, 3, 4)
        table.advance(3);
        assert_eq!(table.seq_len(), 4);
        assert_eq!(table.current_offset(), 0); // block is full
        assert!(table.needs_new_block()); // next token needs new block

        // Allocate second block
        table.append_block(3); // physical block 3
        table.advance(1);
        assert_eq!(table.seq_len(), 5);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.blocks(), &[7, 3]);
        assert_eq!(table.current_offset(), 1);
        assert!(!table.needs_new_block());
    }

    #[test]
    fn block_table_reset() {
        let mut table = BlockTable::new(16);
        table.append_block(5);
        table.advance(10);
        assert_eq!(table.seq_len(), 10);

        table.reset();
        assert_eq!(table.seq_len(), 0);
        assert_eq!(table.num_blocks(), 0);
        assert!(table.needs_new_block());
    }

    #[test]
    fn block_table_prefill() {
        let mut table = BlockTable::new(16);

        // Prefill 40 tokens → needs ceil(40/16) = 3 blocks
        table.append_block(0);
        table.append_block(1);
        table.append_block(2);
        table.advance(40);

        assert_eq!(table.seq_len(), 40);
        assert_eq!(table.num_blocks(), 3);
        assert_eq!(table.current_offset(), 8); // 40 % 16 = 8
        assert!(!table.needs_new_block());

        // Next 8 tokens fill block 2
        table.advance(8);
        assert_eq!(table.seq_len(), 48);
        assert_eq!(table.current_offset(), 0); // block full
        assert!(table.needs_new_block());
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn zero_block_size_panics() {
        BlockAllocator::new(&BlockConfig {
            block_size: 0,
            num_blocks: 10,
        });
    }

    #[test]
    #[should_panic(expected = "num_blocks must be > 0")]
    fn zero_num_blocks_panics() {
        BlockAllocator::new(&BlockConfig {
            block_size: 16,
            num_blocks: 0,
        });
    }
}
