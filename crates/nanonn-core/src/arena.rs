use crate::{AllocError, Allocator};
use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

/// A bump allocator over a caller-provided buffer, typically a `static` array.
///
/// It lets nanonn run without any heap: pass `&arena` wherever an allocator is expected.
/// Memory is reclaimed when the most recent allocation is freed (stack order), or all at once
/// with [`Arena::reset`].
///
/// ```
/// use nanonn_core::{Arena, Allocator};
/// use core::alloc::Layout;
///
/// let mut buffer = [0u8; 256];
/// let arena = Arena::new(&mut buffer);
/// let block = arena.allocate(Layout::from_size_align(64, 16).unwrap()).unwrap();
/// assert_eq!(block.len(), 64);
/// assert!(arena.used() >= 64);
/// ```
pub struct Arena<'a> {
    base: NonNull<u8>,
    capacity: usize,
    used: Cell<usize>,
    _buffer: PhantomData<&'a mut [u8]>,
}

impl<'a> Arena<'a> {
    /// An arena over `buffer`.
    pub fn new(buffer: &'a mut [u8]) -> Self {
        let capacity = buffer.len();
        Self {
            base: NonNull::from(buffer).cast(),
            capacity,
            used: Cell::new(0),
            _buffer: PhantomData,
        }
    }

    /// An arena over an uninitialised buffer.
    pub fn new_uninit(buffer: &'a mut [MaybeUninit<u8>]) -> Self {
        let capacity = buffer.len();
        Self {
            base: NonNull::from(buffer).cast(),
            capacity,
            used: Cell::new(0),
            _buffer: PhantomData,
        }
    }

    /// Bytes handed out so far, including alignment padding.
    pub fn used(&self) -> usize {
        self.used.get()
    }

    /// Total size of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Frees every allocation at once. Requires `&mut self`, so nothing allocated from the
    /// arena can still be borrowed.
    pub fn reset(&mut self) {
        self.used.set(0);
    }
}

// SAFETY: blocks are disjoint sub-slices of the exclusively borrowed buffer, aligned as requested,
// and stay valid until freed or until `reset` (which needs `&mut self`, so no block is borrowed).
unsafe impl Allocator for Arena<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let base = self.base.as_ptr() as usize;
        let start = base.checked_add(self.used.get()).ok_or(AllocError)?;
        let aligned =
            start.checked_add(layout.align() - 1).ok_or(AllocError)? & !(layout.align() - 1);
        let end = aligned.checked_add(layout.size()).ok_or(AllocError)?;
        if end - base > self.capacity {
            return Err(AllocError);
        }
        self.used.set(end - base);
        // SAFETY: `aligned` lies within the buffer (checked above), so the offset is in bounds.
        let ptr = unsafe { self.base.as_ptr().add(aligned - base) };
        // SAFETY: `ptr` is derived from a non-null base and in bounds.
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let offset = ptr.as_ptr() as usize - self.base.as_ptr() as usize;
        if offset + layout.size() == self.used.get() {
            self.used.set(offset);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_and_reclaim() {
        let mut buffer = [0u8; 100];
        let arena = Arena::new(&mut buffer);
        let a = arena
            .allocate(Layout::from_size_align(10, 1).unwrap())
            .unwrap();
        let b = arena
            .allocate(Layout::from_size_align(8, 8).unwrap())
            .unwrap();
        assert_eq!(b.cast::<u8>().as_ptr() as usize % 8, 0);
        let used = arena.used();
        // SAFETY: `b` was allocated by `arena` with this layout.
        unsafe { arena.deallocate(b.cast(), Layout::from_size_align(8, 8).unwrap()) };
        assert!(arena.used() < used);
        assert!(
            arena
                .allocate(Layout::from_size_align(200, 1).unwrap())
                .is_err()
        );
        let _ = a;
    }
}
