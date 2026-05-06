//! Lightweight spin-wait thread pool for latency-critical GEMM/GEMV.
//!
//! Workers spin on a shared generation counter instead of per-worker status
//! flags.  This keeps the spinning read-shared (no cache-line bouncing while
//! idle) and reduces per-dispatch overhead to two shared atomics.
//!
//! # Design
//!
//! - Fixed number of worker threads, spawned once at creation.
//! - A shared `DispatchState` holds the task descriptor and a generation counter.
//! - `dispatch()` writes the task, bumps the generation (Release), then runs
//!   task 0.  Workers spin on the generation (Acquire), execute if their
//!   `worker_id < num_tasks`, and atomically increment a completion counter.
//! - The calling thread also executes one chunk (task 0), so N threads =
//!   1 caller + (N-1) workers.
//! - Workers spin with `hint::spin_loop()` for ~1µs latency vs ~5-10µs for futex.

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// Shared dispatch state, written by the main thread before bumping the
/// generation counter, read by workers after observing the new generation.
///
/// All fields live on one cache line so that the single `generation` Acquire
/// load also pulls in the task descriptor.
#[repr(align(64))]
struct DispatchState {
    /// Monotonically increasing generation counter.  Main bumps this (Release)
    /// to signal new work; workers spin on it (Acquire).
    generation: AtomicU64,
    /// Number of participating threads (including the caller).  Workers with
    /// `worker_id >= num_tasks` skip this round.
    num_tasks: std::sync::atomic::AtomicUsize,
    /// Trampoline function pointer: `fn(*const (), usize, usize)`.
    trampoline: AtomicPtr<()>,
    /// Pointer to the task closure (type-erased `&F`).
    data: AtomicPtr<()>,
}

/// Per-dispatch completion counter, on its own cache line to avoid
/// false-sharing with `DispatchState` (which workers read-share while
/// spinning).
#[repr(align(64))]
struct CompletionState {
    /// Workers atomically increment this after finishing.  Main spins
    /// until it reaches `num_tasks - 1`.
    completed: std::sync::atomic::AtomicUsize,
}

/// A fixed-size spin-wait thread pool using barrier-based synchronisation.
///
/// Workers spin on a shared generation counter rather than per-worker status
/// flags.  This keeps the spinning read-shared (no cache-line bouncing while
/// idle) and reduces the per-dispatch overhead to two atomic operations:
/// one `fetch_add` on `generation` (main) and one `fetch_add` on `completed`
/// (each worker).
///
/// When physical core topology is detected, each worker is pinned to a
/// distinct physical core to avoid HT siblings competing for execution
/// resources. The caller thread (task 0) is also pinned during dispatch.
pub struct SpinPool {
    state: Arc<DispatchState>,
    completion: Arc<CompletionState>,
    workers: Vec<std::thread::JoinHandle<()>>,
    num_threads: usize,
    /// CPU ID to pin the caller thread to during dispatch (core 0).
    /// `None` if topology detection failed.
    caller_core: Option<usize>,
    /// Ensures only one dispatch is active at a time, preventing concurrent
    /// callers from corrupting shared state (e.g., during parallel tests).
    dispatch_lock: std::sync::Mutex<()>,
    /// Set to true to tell workers to exit.
    shutdown: Arc<AtomicU8>,
}

impl SpinPool {
    /// Create a new spin pool with `num_threads` total threads.
    ///
    /// Thread 0 is the calling thread; threads 1..N are spawned workers.
    /// So `num_threads = 4` spawns 3 background threads.
    ///
    /// # Panics
    /// Panics if `num_threads` is 0.
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "SpinPool needs at least 1 thread");

        let state = Arc::new(DispatchState {
            generation: AtomicU64::new(0),
            num_tasks: std::sync::atomic::AtomicUsize::new(0),
            trampoline: AtomicPtr::new(std::ptr::null_mut()),
            data: AtomicPtr::new(std::ptr::null_mut()),
        });
        let completion = Arc::new(CompletionState {
            completed: std::sync::atomic::AtomicUsize::new(0),
        });
        let shutdown = Arc::new(AtomicU8::new(0));

        let physical_cores = detect_physical_cores();
        let caller_core = physical_cores.as_ref().and_then(|c| c.first().copied());

        let mut workers = Vec::with_capacity(num_threads - 1);

        for worker_id in 1..num_threads {
            let st = Arc::clone(&state);
            let comp = Arc::clone(&completion);
            let shut = Arc::clone(&shutdown);
            let pin_core = physical_cores
                .as_ref()
                .and_then(|cores| cores.get(worker_id).copied());
            let handle = std::thread::Builder::new()
                .name(format!("spin-worker-{worker_id}"))
                .spawn(move || {
                    if let Some(core_id) = pin_core {
                        pin_to_core(core_id);
                    }
                    worker_loop(worker_id, &st, &comp, &shut);
                })
                .expect("failed to spawn spin-pool worker");
            workers.push(handle);
        }

        Self {
            state,
            completion,
            workers,
            num_threads,
            caller_core,
            dispatch_lock: std::sync::Mutex::new(()),
            shutdown,
        }
    }

    /// Number of threads in this pool (including the caller thread).
    #[inline]
    #[must_use]
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
        // Type-erased trampoline: casts data pointer back to &F and calls it.
        fn trampoline<F: Fn(usize, usize) + Sync>(data: *const (), task_id: usize, n: usize) {
            // SAFETY: data points to task_fn on the caller's stack, valid because
            // dispatch() blocks until all workers complete.
            unsafe { (&*data.cast::<F>())(task_id, n) };
        }

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

        // Serialize multi-task dispatches to prevent concurrent callers from
        // corrupting shared state. Single-task fast path above is lock-free.
        let _guard = self
            .dispatch_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let tramp: fn(*const (), usize, usize) = trampoline::<F>;
        let data_ptr = (&raw const task_fn).cast::<()>();

        // Pin caller thread to its physical core (once per thread).
        if let Some(core_id) = self.caller_core {
            thread_local! {
                static PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
            }
            PINNED.with(|p| {
                if !p.get() {
                    pin_to_core(core_id);
                    p.set(true);
                }
            });
        }

        // Publish task descriptor, then bump generation to release workers.
        self.state
            .trampoline
            .store(tramp as *mut (), Ordering::Relaxed);
        self.state
            .data
            .store(data_ptr.cast_mut(), Ordering::Relaxed);
        self.state.num_tasks.store(num_tasks, Ordering::Relaxed);
        self.completion.completed.store(0, Ordering::Relaxed);
        // Release: ensures all stores above are visible before workers read them.
        self.state.generation.fetch_add(1, Ordering::Release);

        // Execute task 0 on the calling thread.
        task_fn(0, num_tasks);

        // Wait for all workers to complete.
        let expected = num_tasks - 1;
        while self.completion.completed.load(Ordering::Acquire) != expected {
            std::hint::spin_loop();
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        // Signal shutdown and bump generation so workers see it.
        self.shutdown.store(1, Ordering::Relaxed);
        self.state.generation.fetch_add(1, Ordering::Release);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Worker spin loop: wait for generation change, execute if participating.
fn worker_loop(
    worker_id: usize,
    state: &DispatchState,
    completion: &CompletionState,
    shutdown: &AtomicU8,
) {
    let mut last_gen = 0u64;

    loop {
        // Spin until we see a new generation.
        loop {
            let gen = state.generation.load(Ordering::Acquire);
            if gen != last_gen {
                last_gen = gen;
                break;
            }
            std::hint::spin_loop();
        }

        // Check for shutdown.
        if shutdown.load(Ordering::Relaxed) != 0 {
            return;
        }

        // Check if this worker participates in the current dispatch.
        let num_tasks = state.num_tasks.load(Ordering::Relaxed);
        if worker_id >= num_tasks {
            // Not participating — go back to spinning on generation.
            continue;
        }

        // Save the generation we woke up for before executing the task.
        // This is used after the task completes to guard against stale signals.
        let my_gen = last_gen;

        // The Acquire on generation ensures we see the trampoline/data stores.
        let tramp_ptr = state.trampoline.load(Ordering::Relaxed);
        let data_ptr = state.data.load(Ordering::Relaxed);

        // SAFETY: tramp_ptr is a valid fn(*const (), usize, usize),
        // data_ptr points to the caller's stack-local closure reference.
        let tramp: fn(*const (), usize, usize) = unsafe { std::mem::transmute(tramp_ptr) };
        tramp(data_ptr.cast_const(), worker_id, num_tasks);

        // Only signal completion if the dispatch we participated in is still
        // current.  Under heavy preemption a worker can be descheduled between
        // executing the task and reaching this point.  If the main thread has
        // already timed out waiting (impossible with our blocking wait) *or* if
        // a new dispatch has been issued (generation changed), the old worker's
        // fetch_add would corrupt the new round's counter, causing the main
        // thread to exit its spin loop one completion short and return while a
        // worker from the new round is still writing.  The generation check
        // makes such stale workers silently discard their signal — the main
        // thread already exited correctly (it saw `num_tasks - 1` from the real
        // workers), and the new round's counter remains uncorrupted.
        let current_gen = state.generation.load(Ordering::Acquire);
        if current_gen == my_gen {
            completion.completed.fetch_add(1, Ordering::Release);
        }
    }
}

/// Detect one logical CPU ID per physical core from Linux sysfs topology.
///
/// Reads `/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list`
/// and picks the first (lowest-numbered) sibling from each unique group.
/// Returns `None` if the sysfs topology is unavailable (non-Linux, containers, etc.).
fn detect_physical_cores() -> Option<Vec<usize>> {
    use std::collections::BTreeSet;
    use std::fs;

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    if !cpu_dir.exists() {
        return None;
    }

    // Collect unique physical core groups. Each group is identified by its
    // thread_siblings_list (e.g., "0,8" means CPU 0 and CPU 8 share a core).
    // We pick the lowest CPU id from each group.
    let mut seen_groups = BTreeSet::new();
    let mut physical = Vec::new();

    // Iterate CPU indices 0..N until we stop finding them.
    for cpu_id in 0..1024 {
        let path = cpu_dir
            .join(format!("cpu{cpu_id}"))
            .join("topology/thread_siblings_list");
        let Ok(content) = fs::read_to_string(&path) else {
            if cpu_id == 0 {
                return None; // Can't even read cpu0 — give up
            }
            break; // Past last CPU
        };
        let siblings = content.trim().to_string();
        if seen_groups.insert(siblings) {
            // First time seeing this sibling group — cpu_id is the representative
            physical.push(cpu_id);
        }
    }

    if physical.is_empty() {
        None
    } else {
        Some(physical)
    }
}

/// Pin the calling thread to a specific logical CPU.
///
/// Uses `sched_setaffinity` on Linux. Silently does nothing on failure
/// or non-Linux platforms.
fn pin_to_core(core_id: usize) {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(core_id, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw const set);
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = core_id;
    }
}

/// Global spin pool, lazily initialized.
///
/// Returns a reference to a shared `SpinPool` with one thread per physical core.
/// The pool is created once and reused for all GEMV calls.
///
/// Thread count selection (in order of priority):
/// 1. `RAYON_NUM_THREADS` env var (explicit override)
/// 2. Linux sysfs topology → physical core count, capped by cgroup limit
/// 3. macOS `sysctl hw.physicalcpu`
/// 4. `available_parallelism() / 2` (assume HT, conservative)
pub fn global_pool() -> &'static SpinPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<SpinPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(default_thread_count);
        SpinPool::new(n)
    })
}

/// Determine the optimal thread count for this machine.
///
/// Uses `num_cpus::get_physical()` for cross-platform physical core detection
/// (Linux, macOS, Windows, FreeBSD), capped by `available_parallelism()` which
/// respects cgroup CPU limits in containers. Without the cap, a container with
/// `--cpus=4` would detect all host physical cores and over-subscribe.
fn default_thread_count() -> usize {
    let physical = num_cpus::get_physical();
    let cgroup_limit =
        std::thread::available_parallelism().map_or(physical, std::num::NonZero::get);
    physical.min(cgroup_limit).max(1)
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

    /// Stress test for the generation-keyed completion race.
    ///
    /// Without the generation check in `worker_loop`, a worker preempted
    /// between "execute task" and "fetch_add(1)" can fire its stale
    /// increment into the *next* dispatch's counter, causing the main thread
    /// to exit the spin loop one completion short. That manifests here as a
    /// sum that doesn't equal the expected value — proving a worker was still
    /// running when `dispatch` returned.
    ///
    /// Run with many threads and many iterations to stress the timing window.
    #[test]
    fn test_repeated_dispatch_race_stress() {
        let num_threads = (std::thread::available_parallelism().map_or(4, std::num::NonZero::get))
            .min(16)
            .max(2);
        let pool = SpinPool::new(num_threads);
        let expected: u64 = (0..num_threads as u64).sum(); // 0+1+…+(N-1)

        for iteration in 0..500 {
            let sum = AtomicU64::new(0);
            pool.dispatch(num_threads, |task_id, _| {
                // Simulate a tiny amount of work so tasks finish at slightly
                // different times, increasing the probability of catching a
                // worker that is slow to signal completion.
                std::hint::spin_loop();
                sum.fetch_add(task_id as u64, Ordering::Relaxed);
            });
            let got = sum.load(Ordering::Relaxed);
            assert_eq!(
                got, expected,
                "iteration {iteration}: sum={got} expected={expected} (race detected)"
            );
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
