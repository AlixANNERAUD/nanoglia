//! The lock guarding ESP-NN's global state.
//!
//! ESP-NN keeps the scratch buffers of its kernels in global variables, set by
//! `esp_nn_set_*_scratch_buf` right before the kernel call that uses them: ESP-NN is not
//! reentrant. Two contexts (threads, tasks, interrupt handlers, cores) running ESP-NN at the same
//! time would overwrite each other's scratch pointer. The lock is a try-lock on an atomic flag, so
//! it never blocks and needs no critical section: a context that does not get it runs another
//! implementation instead.

#[cfg(target_has_atomic = "8")]
use core::sync::atomic::{AtomicBool, Ordering};

/// Whether [`try_lock`] can ever succeed: the target has atomic compare-and-swap on bytes.
/// Without it (Cortex-M0, `riscv32imc`...) the lock always reports contention.
pub const LOCK_AVAILABLE: bool = cfg!(target_has_atomic = "8");

#[cfg(target_has_atomic = "8")]
static LOCKED: AtomicBool = AtomicBool::new(false);

/// Exclusive access to ESP-NN's global scratch buffers, released on drop.
///
/// Hold one from the `set_*_scratch_buf` call to the end of the kernel call that reads the
/// buffer. Code calling ESP-NN directly next to nanoglia must take it too, so that nanoglia's
/// kernels fall back to another implementation while it runs.
#[derive(Debug)]
#[must_use = "the lock is released when the guard is dropped"]
pub struct ScratchLock {
    _private: (),
}

/// Takes the ESP-NN lock if it is free; `None` if another context holds it (or if the target has
/// no atomic compare-and-swap, see [`LOCK_AVAILABLE`]). Never blocks.
pub fn try_lock() -> Option<ScratchLock> {
    #[cfg(target_has_atomic = "8")]
    {
        LOCKED
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| ScratchLock { _private: () })
    }
    #[cfg(not(target_has_atomic = "8"))]
    {
        None
    }
}

impl Drop for ScratchLock {
    fn drop(&mut self) {
        #[cfg(target_has_atomic = "8")]
        LOCKED.store(false, Ordering::Release);
    }
}
