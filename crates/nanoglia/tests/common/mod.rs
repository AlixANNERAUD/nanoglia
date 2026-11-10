//! Helpers shared by the integration tests.

use std::path::{Path, PathBuf};

/// The repository root.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Model bytes copied into a 16-byte aligned buffer, as firmware would place them.
pub struct Aligned {
    storage: Vec<u128>,
    len: usize,
}

impl Aligned {
    pub fn read(path: &Path) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Self::copy(&bytes)
    }

    #[allow(dead_code)]
    pub fn copy(bytes: &[u8]) -> Self {
        let mut storage = vec![0u128; bytes.len().div_ceil(16)];
        // SAFETY: `storage` holds at least `bytes.len()` bytes.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), bytes.len())
        };
        dst.copy_from_slice(bytes);
        Self {
            storage,
            len: bytes.len(),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the first `len` bytes were initialised from the file.
        unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast::<u8>(), self.len) }
    }
}
