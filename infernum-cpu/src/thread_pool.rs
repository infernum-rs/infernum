//! Lightweight spin-wait thread pool for latency-critical GEMV.
//!
//! Workers spin on an atomic flag instead of sleeping through futex,
//! eliminating kernel round-trips and keeping the memory prefetch pipeline warm.
//!
//! # Design
//!
//! - Fixed number of worker threads, spawned once at creation.
//! - Each worker has its own `WorkerSlot` containing an atomic task pointer
//!   and status flag.
//! - `dispatch()` writes task descriptors and sets each worker's flag to `Ready`.
//!   Workers spin until they see `Ready`, execute, then set `Done`.
//! - The calling thread also executes one chunk (slot 0), so N threads =
//!   1 caller + (N-1) workers.
//! - Workers spin with `hint::spin_loop()` for ~1µs latency vs ~5-10µs for futex.

use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};
use std::sync::Arc;

/// Status values for worker slots.
const IDLE: u8 = 0;
const READY: u8 = 1;
const DONE: u8 = 2;
const SHUTDOWN: u8 = 3;

/// Per-worker communication slot (cache-line padded).
///
/// The task is transmitted as a type-erased function call: a trampoline
/// function pointer plus a data pointer. The trampoline casts the data
/// pointer back to `&F` and calls `F(task_id, num_tasks)`.
#[repr(align(64))]
struct WorkerSlot {
    /// Status flag: IDLE → READY (main sets) → DONE (worker sets).
    status: AtomicU8,
    /// Trampoline function pointer: `fn(*const (), usize, usize)`.
    trampoline: AtomicPtr<()>,
    /// Pointer to the task closure (type-erased `&F`).
    data: AtomicPtr<()>,
    /// This worker's task index.
    task_id: std::sync::atomic::AtomicUsize,
    /// Total number of tasks in this dispatch.
    num_tasks: std::sync::atomic::AtomicUsize,
}

impl WorkerSlot {
    fn new() -> Self {
        Self {
            status: AtomicU8::new(IDLE),
            trampoline: AtomicPtr::new(std::ptr::null_mut()),
            data: AtomicPtr::new(std::ptr::null_mut()),
            task_id: std::sync::atomic::AtomicUsize::new(0),
            num_tasks: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

/// A fixed-size spin-wait thread pool.
///
/// Workers spin on atomic flags rather than sleeping via futex/condvar.
/// This trades CPU usage for latency — workers burn cycles when idle,
/// but wake up in ~1µs instead of ~5-10µs.
pub struct SpinPool {
    slots: Arc<Vec<WorkerSlot>>,
    _workers: Vec<std::thread::JoinHandle<()>>,
    num_threads: usize,
}

impl SpinPool {
    /// Create a new spin pool with `num_threads` total threads.
    ///
    /// Thread 0 is the calling thread; threads 1..N are spawned workers.
    /// So `num_threads = 4` spawns 3 background threads.
    ///
    /// # Panics
    /// Panics if `num_threads` is 0.
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "SpinPool needs at least 1 thread");

        let mut slots = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            slots.push(WorkerSlot::new());
        }
        let slots = Arc::new(slots);

        let mut workers = Vec::with_capacity(num_threads - 1);

        for worker_id in 1..num_threads {
            let slots_clone = Arc::clone(&slots);
            let handle = std::thread::Builder::new()
                .name(format!("spin-worker-{worker_id}"))
                .spawn(move || {
                    worker_loop(&slots_clone[worker_id]);
                })
                .expect("failed to spawn spin-pool worker");
            workers.push(handle);
        }

        Self {
            slots,
            _workers: workers,
            num_threads,
        }
    }

    /// Number of threads in this pool (including the caller thread).
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Execute `num_tasks` independent tasks in parallel.
    ///
    /// The `task_fn` is called with `(task_index, num_tasks)` for each task.
    /// Task 0 runs on the calling thread; tasks 1..N run on workers.
    /// Blocks until all tasks complete.
    ///
    /// # Panics
    /// Panics if `num_tasks > num_threads`.
    pub fn dispatch<F>(&self, num_tasks: usize, task_fn: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        assert!(
            num_tasks <= self.num_threads,
            "dispatch: num_tasks ({num_tasks}) > num_threads ({})",
            self.num_threads
        );

        if num_tasks == 0 {
            return;
        }

        if num_tasks == 1 {
            task_fn(0, 1);
            return;
        }

        // Type-erased trampoline: casts data pointer back to &F and calls it.
        fn trampoline<F: Fn(usize, usize) + Sync>(data: *const (), task_id: usize, n: usize) {
            // SAFETY: data points to task_fn on the caller's stack, valid because
            // dispatch() blocks until all workers complete.
            unsafe { (&*data.cast::<F>())(task_id, n) };
        }

        let tramp: fn(*const (), usize, usize) = trampoline::<F>;
        let data_ptr = (&task_fn as *const F).cast::<()>();

        for worker_id in 1..num_tasks {
            self.slots[worker_id]
                .trampoline
                .store(tramp as *mut (), Ordering::Relaxed);
            self.slots[worker_id]
                .data
                .store(data_ptr as *mut (), Ordering::Relaxed);
            self.slots[worker_id]
                .task_id
                .store(worker_id, Ordering::Relaxed);
            self.slots[worker_id]
                .num_tasks
                .store(num_tasks, Ordering::Relaxed);
            // Release fence: makes all preceding stores visible to the worker.
            self.slots[worker_id].status.store(READY, Ordering::Release);
        }

        // Execute task 0 on the calling thread.
        task_fn(0, num_tasks);

        // Wait for all workers to complete.
        for worker_id in 1..num_tasks {
            while self.slots[worker_id].status.load(Ordering::Acquire) != DONE {
                std::hint::spin_loop();
            }
            // Reset to IDLE for next dispatch.
            self.slots[worker_id].status.store(IDLE, Ordering::Release);
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        // Signal all workers to shut down.
        for slot in self.slots.iter().skip(1) {
            slot.status.store(SHUTDOWN, Ordering::Release);
        }
        // Workers will see SHUTDOWN and exit their loop.
        // Join is handled by JoinHandle drop, but we drain explicitly to catch panics.
        for handle in self._workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Worker spin loop: wait for READY, execute task, set DONE.
fn worker_loop(slot: &WorkerSlot) {
    loop {
        // Spin until we see READY or SHUTDOWN.
        loop {
            let status = slot.status.load(Ordering::Acquire);
            if status == READY {
                break;
            }
            if status == SHUTDOWN {
                return;
            }
            std::hint::spin_loop();
        }

        // The Acquire on status ensures we see the trampoline/data stores.
        let tramp_ptr = slot.trampoline.load(Ordering::Relaxed);
        let data_ptr = slot.data.load(Ordering::Relaxed);
        let task_id = slot.task_id.load(Ordering::Relaxed);
        let num_tasks = slot.num_tasks.load(Ordering::Relaxed);

        // SAFETY: tramp_ptr is a valid fn(*const (), usize, usize),
        // data_ptr points to the caller's stack-local closure reference.
        let tramp: fn(*const (), usize, usize) = unsafe { std::mem::transmute(tramp_ptr) };
        tramp(data_ptr.cast_const(), task_id, num_tasks);

        // Signal completion.
        slot.status.store(DONE, Ordering::Release);
    }
}

/// Global spin pool, lazily initialized.
///
/// Returns a reference to a shared `SpinPool` with one thread per physical core.
/// The pool is created once and reused for all GEMV calls.
pub fn global_pool() -> &'static SpinPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<SpinPool> = OnceLock::new();
    POOL.get_or_init(|| {
        // Use RAYON_NUM_THREADS if set (for consistency with benchmarks),
        // otherwise use available parallelism (usually physical core count).
        let n = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(1)
            });
        SpinPool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_single_thread() {
        let pool = SpinPool::new(1);
        let sum = AtomicU64::new(0);
        pool.dispatch(1, |task_id, _| {
            sum.fetch_add(task_id as u64 + 1, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_multi_thread_sum() {
        let pool = SpinPool::new(4);
        let sum = AtomicU64::new(0);
        pool.dispatch(4, |task_id, _| {
            sum.fetch_add(task_id as u64 + 1, Ordering::Relaxed);
        });
        // 1 + 2 + 3 + 4 = 10
        assert_eq!(sum.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_repeated_dispatch() {
        let pool = SpinPool::new(4);
        for _ in 0..100 {
            let sum = AtomicU64::new(0);
            pool.dispatch(4, |task_id, _| {
                sum.fetch_add(task_id as u64, Ordering::Relaxed);
            });
            // 0 + 1 + 2 + 3 = 6
            assert_eq!(sum.load(Ordering::Relaxed), 6);
        }
    }

    #[test]
    fn test_dispatch_fewer_tasks_than_threads() {
        let pool = SpinPool::new(8);
        let sum = AtomicU64::new(0);
        pool.dispatch(3, |task_id, _| {
            sum.fetch_add(task_id as u64 + 1, Ordering::Relaxed);
        });
        // 1 + 2 + 3 = 6
        assert_eq!(sum.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn test_parallel_writes() {
        let pool = SpinPool::new(4);
        let data = vec![0u64; 1000];
        let data_ptr = data.as_ptr() as usize;
        let data_len = data.len();

        pool.dispatch(4, |task_id, num_tasks| {
            let chunk_size = (data_len + num_tasks - 1) / num_tasks;
            let start = task_id * chunk_size;
            let end = (start + chunk_size).min(data_len);
            // SAFETY: each task writes to a non-overlapping slice
            let ptr = data_ptr as *mut u64;
            for i in start..end {
                unsafe { *ptr.add(i) = (task_id + 1) as u64 };
            }
        });

        // Verify all elements were written
        for &val in &data {
            assert!(val >= 1 && val <= 4, "unexpected value: {val}");
        }
    }
}
