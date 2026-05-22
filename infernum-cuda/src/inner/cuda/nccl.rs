//! NCCL communicator for multi-GPU tensor parallelism
//!
//! Thin wrapper around `cudarc::nccl` providing an ergonomic API for
//! all-reduce operations on `CudaTensor`s.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use cudarc::nccl::safe::{Comm, Id, ReduceOp};
use half::bf16;

use crate::cuda::CudaTensor;
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

/// Wraps an NCCL communicator with convenience methods for tensor operations.
///
/// Each GPU thread holds one `NcclCommunicator`. All communicators in a group
/// must participate in collective operations simultaneously.
pub struct NcclCommunicator {
    comm: Comm,
}

// SAFETY: NcclCommunicator is Send because we own the comm handle and each
// GPU thread gets its own communicator. The raw *mut ncclComm prevents
// auto-derive but the handle is safe to move between threads.
unsafe impl Send for NcclCommunicator {}

// SAFETY: NcclCommunicator is Sync because all methods take &self and NCCL
// serializes access internally. In practice we only access from one thread.
unsafe impl Sync for NcclCommunicator {}

impl NcclCommunicator {
    /// Create communicators for all available GPUs on a single node.
    ///
    /// Returns one `NcclCommunicator` per GPU, ordered by device ordinal.
    /// All communicators share the same NCCL group.
    ///
    /// # Errors
    /// Returns an error if NCCL initialization fails.
    pub fn from_devices(devices: Vec<Arc<CudaDevice>>) -> Result<Vec<Self>> {
        let comms = Comm::from_devices(devices)?;
        Ok(comms.into_iter().map(|comm| Self { comm }).collect())
    }

    /// Create a single communicator for one rank in a multi-process/thread group.
    ///
    /// Rank 0 should generate the `Id` via [`NcclId::new()`] and broadcast it
    /// to all other ranks before calling this.
    ///
    /// # Errors
    /// Returns an error if NCCL initialization fails.
    pub fn from_rank(
        device: Arc<CudaDevice>,
        rank: usize,
        world_size: usize,
        id: NcclId,
    ) -> Result<Self> {
        let comm = Comm::from_rank(device, rank, world_size, id.0)?;
        Ok(Self { comm })
    }

    /// The CUDA device this communicator was created on.
    #[must_use]
    pub fn device(&self) -> Arc<CudaDevice> {
        self.comm.device()
    }

    /// This communicator's rank (0-based).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.comm.rank()
    }

    /// Total number of GPUs in the group.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.comm.world_size()
    }

    /// In-place sum-reduce a tensor across all ranks.
    ///
    /// After this call, every rank holds the element-wise sum of all inputs.
    /// The operation is asynchronous on the GPU stream; call `ctx.synchronize()`
    /// if you need the result on the host.
    ///
    /// # Errors
    /// Returns an error if the NCCL all-reduce fails.
    pub fn all_reduce_sum_inplace(&self, tensor: &mut CudaTensor) -> Result<()> {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        let numel = tensor.numel();
        let device = tensor.context().device().clone();

        // Allocate a temporary output buffer for the reduction
        let recv = unsafe { CudaTensor::uninit(tensor.context(), &shape, dtype)? };

        // NCCL AllReduce must use the correct element type so it performs
        // floating-point summation, not byte-wise integer summation.
        // `cuda_slice()` returns raw `CudaSlice<u8>`, which would cause NCCL
        // to treat BF16/F32 values as u8 bytes and produce wrong results.
        // We reinterpret the buffers as typed slices via `upgrade_device_ptr`
        // and std::mem::forget them afterward to prevent double-free.
        match dtype {
            DType::BF16 => {
                let send_ptr = tensor.device_ptr();
                let recv_ptr = recv.device_ptr();
                let send_typed = unsafe { device.upgrade_device_ptr::<bf16>(send_ptr, numel) };
                let mut recv_typed = unsafe { device.upgrade_device_ptr::<bf16>(recv_ptr, numel) };
                self.comm
                    .all_reduce(&send_typed, &mut recv_typed, &ReduceOp::Sum)?;
                std::mem::forget(send_typed);
                std::mem::forget(recv_typed);
            }
            DType::F32 => {
                let send_ptr = tensor.device_ptr();
                let recv_ptr = recv.device_ptr();
                let send_typed = unsafe { device.upgrade_device_ptr::<f32>(send_ptr, numel) };
                let mut recv_typed = unsafe { device.upgrade_device_ptr::<f32>(recv_ptr, numel) };
                self.comm
                    .all_reduce(&send_typed, &mut recv_typed, &ReduceOp::Sum)?;
                std::mem::forget(send_typed);
                std::mem::forget(recv_typed);
            }
            other => {
                return Err(infernum::Error::UnsupportedDtype(format!(
                    "all_reduce_sum_inplace: unsupported dtype {other:?}"
                )))
            }
        }

        // Swap the result back into the original tensor
        *tensor = recv;
        Ok(())
    }
}

impl infernum::Comm<CudaTensor> for NcclCommunicator {
    fn all_reduce_sum(&self, tensor: &mut CudaTensor) -> Result<()> {
        self.all_reduce_sum_inplace(tensor)
    }
}

/// Opaque NCCL unique ID for coordinating communicator creation.
///
/// Rank 0 creates this and broadcasts the raw bytes to other ranks.
#[derive(Debug, Clone, Copy)]
pub struct NcclId(Id);

impl NcclId {
    /// Generate a new unique ID (call on rank 0 only).
    ///
    /// # Errors
    /// Returns an error if NCCL ID generation fails.
    pub fn new() -> Result<Self> {
        Ok(Self(Id::new()?))
    }

    /// Reconstruct from raw bytes (received from rank 0).
    #[must_use]
    pub fn from_raw(internal: [core::ffi::c_char; 128]) -> Self {
        Self(Id::uninit(internal))
    }

    /// Get the raw bytes for transmission to other ranks.
    #[must_use]
    pub fn to_raw(&self) -> &[core::ffi::c_char; 128] {
        self.0.internal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_nccl_id_roundtrip() {
        let id = NcclId::new().expect("Failed to create NCCL ID");
        let raw = *id.to_raw();
        let id2 = NcclId::from_raw(raw);
        assert_eq!(id.to_raw(), id2.to_raw());
    }

    // Helper: spawn N threads each with their own CudaContext + NcclComm sharing the same
    // underlying CudaDevice. Matches how ShardedGraphEngine::from_config wires things up.
    fn run_all_reduce_test(n_devices: usize, id: NcclId) -> Vec<std::thread::JoinHandle<Vec<f32>>> {
        (0..n_devices)
            .map(|rank| {
                let id_raw = *id.to_raw();
                std::thread::spawn(move || {
                    // Create context (with non-default stream) and NCCL comm from the same device.
                    let ctx = CudaContext::new(rank).unwrap();
                    let comm_id = NcclId::from_raw(id_raw);
                    let comm = NcclCommunicator::from_rank(
                        std::sync::Arc::clone(ctx.device()),
                        rank,
                        n_devices,
                        comm_id,
                    )
                    .unwrap();

                    let data = vec![(rank + 1) as f32; 4];
                    let mut tensor = CudaTensor::from_slice(&ctx, &[4], &data).unwrap();

                    comm.all_reduce_sum_inplace(&mut tensor).unwrap();
                    ctx.synchronize().unwrap();

                    tensor.to_vec::<f32>().unwrap()
                })
            })
            .collect()
    }

    #[test]
    fn test_nccl_all_reduce_sum_f32() {
        let n_devices = CudaDevice::count().unwrap() as usize;
        if n_devices < 2 {
            eprintln!("Skipping multi-GPU test: only {n_devices} device(s) available");
            return;
        }

        let id = NcclId::new().unwrap();
        let expected_sum = (1..=n_devices).sum::<usize>() as f32;
        let handles = run_all_reduce_test(n_devices, id);

        for (rank, h) in handles.into_iter().enumerate() {
            let result = h.join().unwrap();
            for val in &result {
                assert!(
                    (val - expected_sum).abs() < 1e-3,
                    "Rank {rank}: expected {expected_sum}, got {val}"
                );
            }
        }
    }

    #[test]
    fn test_nccl_all_reduce_sum_bf16() {
        use half::bf16;

        let n_devices = CudaDevice::count().unwrap() as usize;
        if n_devices < 2 {
            eprintln!("Skipping multi-GPU test: only {n_devices} device(s) available");
            return;
        }

        let id = NcclId::new().unwrap();
        let expected_sum = (1..=n_devices).sum::<usize>() as f32;

        let handles: Vec<_> = (0..n_devices)
            .map(|rank| {
                let id_raw = *id.to_raw();
                std::thread::spawn(move || {
                    let ctx = CudaContext::new(rank).unwrap();
                    let comm_id = NcclId::from_raw(id_raw);
                    let comm = NcclCommunicator::from_rank(
                        std::sync::Arc::clone(ctx.device()),
                        rank,
                        n_devices,
                        comm_id,
                    )
                    .unwrap();

                    let data: Vec<bf16> = vec![bf16::from_f32((rank + 1) as f32); 4];
                    let mut tensor = CudaTensor::from_slice(&ctx, &[4], &data).unwrap();

                    comm.all_reduce_sum_inplace(&mut tensor).unwrap();
                    ctx.synchronize().unwrap();

                    tensor.to_vec::<bf16>().unwrap()
                })
            })
            .collect();

        for (rank, h) in handles.into_iter().enumerate() {
            let result = h.join().unwrap();
            for val in &result {
                let val_f32 = val.to_f32();
                assert!(
                    (val_f32 - expected_sum).abs() < 1.0,
                    "Rank {rank}: expected {expected_sum}, got {val_f32}"
                );
            }
        }
    }
}
