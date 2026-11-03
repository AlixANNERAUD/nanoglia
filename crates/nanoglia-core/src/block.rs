use crate::{ALIGN, Allocator, Error, Result};
use core::alloc::Layout;
use core::ptr::NonNull;

/// A zeroed, [`ALIGN`]-aligned byte buffer owned through an allocator and freed on drop.
///
/// This is the only kind of allocation nanoglia performs.
pub struct Block<A: Allocator> {
    ptr: NonNull<u8>,
    len: usize,
    alloc: A,
}

// SAFETY: `Block` owns its memory like a `Box<[u8], A>`.
unsafe impl<A: Allocator + Send> Send for Block<A> {}
// SAFETY: shared access only hands out `&[u8]`.
unsafe impl<A: Allocator + Sync> Sync for Block<A> {}

impl<A: Allocator> Block<A> {
    /// Allocates `len` zeroed bytes from `alloc`. A zero-length block does not allocate.
    pub fn new_zeroed(len: usize, alloc: A) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::<[u8; ALIGN]>::dangling().cast(),
                len,
                alloc,
            });
        }
        let layout = Layout::from_size_align(len, ALIGN).map_err(|_| Error::out_of_memory(len))?;
        let ptr = alloc
            .allocate_zeroed(layout)
            .map_err(|_| Error::out_of_memory(len))?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
            alloc,
        })
    }

    /// The contents.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` points to `len` initialised (zeroed) bytes owned by `self`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The contents, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` guarantees uniqueness.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// A raw pointer to the contents, for building disjoint sub-slices.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<A: Allocator> Drop for Block<A> {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated in `new_zeroed` from `self.alloc` with this exact layout.
            unsafe {
                self.alloc
                    .deallocate(self.ptr, Layout::from_size_align_unchecked(self.len, ALIGN));
            }
        }
    }
}
