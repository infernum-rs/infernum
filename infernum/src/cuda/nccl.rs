//! NCCL communicator for multi-GPU tensor parallelism
//!
//! Thin wrapper around `cudarc::nccl` providing an ergonomic API for
//! all-reduce operations on `CudaTensor`s.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use cudarc::nccl::safe::{Comm, Id, NcclType, ReduceOp};

use crate::cuda::CudaTensor;
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

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
    pub fn all_reduce_sum_inplace<T>(&self, tensor: &mut CudaTensor<T>) -> Result<()>
    where
        T: TensorDType + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + NcclType,
    {
        let shape = tensor.shape().to_vec();

        // Allocate a temporary output buffer for the reduction
        let mut recv = unsafe { CudaTensor::<T>::uninit(tensor.context(), &shape)? };

        let send = tensor.cuda_slice();
        self.comm
            .all_reduce(&send, recv.cuda_slice_mut(), &ReduceOp::Sum)?;

        // Swap the result back into the original tensor
        *tensor = recv;
        Ok(())
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

    #[test]
    fn test_nccl_all_reduce_sum() {
        let n_devices = CudaDevice::count().unwrap() as usize;
        if n_devices < 2 {
            eprintln!("Skipping multi-GPU test: only {n_devices} device(s) available");
            return;
        }

        let devices: Vec<_> = (0..n_devices)
            .map(|i| CudaDevice::new(i).unwrap())
            .collect();
        let comms = NcclCommunicator::from_devices(devices).unwrap();

        std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    s.spawn(move || {
                        let ctx = CudaContext::new(rank).unwrap();
                        let data = vec![(rank + 1) as f32; 4];
                        let mut tensor = CudaTensor::from_slice(&ctx, &[4], &data).unwrap();

                        comm.all_reduce_sum_inplace(&mut tensor).unwrap();
                        ctx.synchronize().unwrap();

                        let result = tensor.to_vec().unwrap();
                        let expected = (1..=n_devices).sum::<usize>() as f32;
                        for val in &result {
                            assert!(
                                (val - expected).abs() < 1e-5,
                                "Expected {expected}, got {val}"
                            );
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    }
}
