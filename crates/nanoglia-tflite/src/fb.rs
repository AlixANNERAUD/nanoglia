//! Minimal flatbuffer primitives: tables, vtables, vectors, strings and unions.
//!
//! Every read is bounds-checked and returns [`Error::malformed`] with the offending offset
//! instead of panicking, so these views stay safe on unverified bytes. Scalars are decoded
//! with [`nanoglia_core::le`], without any alignment requirement.

use nanoglia_core::Error;
use nanoglia_core::Result;
use nanoglia_core::le::{self, LeScalar, LeSlice};

/// Reads a scalar at `pos`.
#[inline]
pub(crate) fn read<T: LeScalar>(buf: &[u8], pos: usize) -> Result<T> {
    le::read(buf, pos).ok_or(Error::malformed(pos))
}

/// Follows the `uoffset_t` stored at `pos` and returns the position it points to.
pub(crate) fn follow(buf: &[u8], pos: usize) -> Result<usize> {
    let offset: u32 = read(buf, pos)?;
    // Offsets are signed-positive in flatbuffers (buffers are limited to 2 GiB).
    if offset > i32::MAX as u32 {
        return Err(Error::malformed(pos));
    }
    let target = pos
        .checked_add(offset as usize)
        .ok_or(Error::malformed(pos))?;
    if target >= buf.len() {
        return Err(Error::malformed(pos));
    }
    Ok(target)
}

/// A vector: a `u32` length followed by `len` elements of `elem_size` bytes.
#[derive(Clone, Copy)]
pub(crate) struct Vector<'m> {
    buf: &'m [u8],
    /// Position of the first element.
    start: usize,
    len: usize,
}

impl<'m> Vector<'m> {
    /// Checks the vector at `pos` and that its `len * elem_size` bytes are in bounds.
    pub(crate) fn at(buf: &'m [u8], pos: usize, elem_size: usize) -> Result<Self> {
        let len: u32 = read(buf, pos)?;
        let len = len as usize;
        let start = pos + 4; // `read` succeeded, so `pos + 4 <= buf.len()`.
        let end = len
            .checked_mul(elem_size)
            .and_then(|n| n.checked_add(start))
            .ok_or(Error::malformed(pos))?;
        if end > buf.len() {
            return Err(Error::malformed(pos));
        }
        Ok(Self { buf, start, len })
    }

    /// Number of elements.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Position of the length prefix, for error reports.
    #[inline]
    pub(crate) fn pos(&self) -> usize {
        self.start - 4
    }

    /// The element bytes, assuming `elem_size` bytes per element.
    fn bytes(&self, elem_size: usize) -> &'m [u8] {
        // Checked in `at` with the same `elem_size`; never fails in practice.
        self.len
            .checked_mul(elem_size)
            .and_then(|n| n.checked_add(self.start))
            .and_then(|end| self.buf.get(self.start..end))
            .unwrap_or(&[])
    }

    /// Position of element `index`, assuming 4-byte elements.
    fn slot(&self, index: usize) -> Result<usize> {
        if index >= self.len {
            return Err(Error::malformed(self.pos()));
        }
        Ok(self.start + index * 4)
    }
}

/// A vector of tables (a vector of offsets to tables).
#[derive(Clone, Copy)]
pub(crate) struct Tables<'m>(Vector<'m>);

impl<'m> Tables<'m> {
    /// An empty vector.
    pub(crate) const EMPTY: Self = Self(Vector {
        buf: &[],
        start: 4,
        len: 0,
    });

    /// Number of tables.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// The table at `index`.
    pub(crate) fn get(&self, index: usize) -> Result<Table<'m>> {
        let slot = self.0.slot(index)?;
        Table::at(self.0.buf, follow(self.0.buf, slot)?)
    }
}

/// A table: an `soffset_t` to its vtable followed by its inline fields.
#[derive(Clone, Copy)]
pub(crate) struct Table<'m> {
    buf: &'m [u8],
    pos: usize,
    vtable: usize,
    vtable_len: usize,
    inline_len: usize,
}

impl<'m> Table<'m> {
    /// Checks the table at `pos`: its vtable and its inline fields must be in bounds.
    pub(crate) fn at(buf: &'m [u8], pos: usize) -> Result<Self> {
        let soffset: i32 = read(buf, pos)?;
        let vtable = i64::try_from(pos)
            .ok()
            .and_then(|p| p.checked_sub(i64::from(soffset)))
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(Error::malformed(pos))?;
        let vtable_len = usize::from(read::<u16>(buf, vtable)?);
        let inline_len = usize::from(read::<u16>(buf, vtable + 2)?);
        if vtable_len < 4 || vtable_len % 2 != 0 || vtable + vtable_len > buf.len() {
            return Err(Error::malformed(vtable));
        }
        if inline_len < 4 || pos + inline_len > buf.len() {
            return Err(Error::malformed(vtable));
        }
        Ok(Self {
            buf,
            pos,
            vtable,
            vtable_len,
            inline_len,
        })
    }

    /// Position of the table, for error reports.
    #[inline]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    /// The model bytes this table belongs to.
    #[inline]
    pub(crate) fn buf(&self) -> &'m [u8] {
        self.buf
    }

    /// Position of field `id`, which must hold `size` bytes, or `None` if it is absent.
    pub(crate) fn field(&self, id: u16, size: usize) -> Result<Option<usize>> {
        let slot = 4 + 2 * usize::from(id);
        if slot + 2 > self.vtable_len {
            return Ok(None);
        }
        let voffset = usize::from(read::<u16>(self.buf, self.vtable + slot)?);
        if voffset == 0 {
            return Ok(None);
        }
        if voffset + size > self.inline_len {
            return Err(Error::malformed(self.vtable + slot));
        }
        Ok(Some(self.pos + voffset))
    }

    /// The scalar field `id`, or `default` if it is absent.
    pub(crate) fn scalar<T: LeScalar>(&self, id: u16, default: T) -> Result<T> {
        match self.field(id, T::SIZE)? {
            Some(pos) => read(self.buf, pos),
            None => Ok(default),
        }
    }

    /// The `bool` field `id`, or `default` if it is absent.
    pub(crate) fn flag(&self, id: u16, default: bool) -> Result<bool> {
        Ok(self.scalar::<u8>(id, u8::from(default))? != 0)
    }

    /// Whether the offset field `id` is present.
    pub(crate) fn has(&self, id: u16) -> Result<bool> {
        Ok(self.field(id, 4)?.is_some())
    }

    /// Follows the offset field `id`.
    fn target(&self, id: u16) -> Result<Option<usize>> {
        match self.field(id, 4)? {
            Some(pos) => follow(self.buf, pos).map(Some),
            None => Ok(None),
        }
    }

    /// The table field `id`.
    pub(crate) fn table(&self, id: u16) -> Result<Option<Table<'m>>> {
        match self.target(id)? {
            Some(pos) => Table::at(self.buf, pos).map(Some),
            None => Ok(None),
        }
    }

    /// The vector field `id` of `elem_size`-byte elements.
    pub(crate) fn vector(&self, id: u16, elem_size: usize) -> Result<Option<Vector<'m>>> {
        match self.target(id)? {
            Some(pos) => Vector::at(self.buf, pos, elem_size).map(Some),
            None => Ok(None),
        }
    }

    /// The `[ubyte]` field `id`, empty if absent.
    pub(crate) fn bytes(&self, id: u16) -> Result<&'m [u8]> {
        Ok(self.vector(id, 1)?.map_or(&[][..], |v| v.bytes(1)))
    }

    /// The scalar vector field `id`, empty if absent.
    pub(crate) fn slice<T: LeScalar>(&self, id: u16) -> Result<LeSlice<'m, T>> {
        Ok(self
            .vector(id, T::SIZE)?
            .map_or(LeSlice::EMPTY, |v| LeSlice::new(v.bytes(T::SIZE))))
    }

    /// The table vector field `id`, empty if absent.
    pub(crate) fn tables(&self, id: u16) -> Result<Tables<'m>> {
        Ok(self.vector(id, 4)?.map_or(Tables::EMPTY, Tables))
    }

    /// The string field `id`: UTF-8 and NUL-terminated.
    pub(crate) fn str(&self, id: u16) -> Result<Option<&'m str>> {
        let Some(vector) = self.vector(id, 1)? else {
            return Ok(None);
        };
        let end = vector.start + vector.len;
        if self.buf.get(end) != Some(&0) {
            return Err(Error::malformed(vector.pos()));
        }
        core::str::from_utf8(vector.bytes(1))
            .map(Some)
            .map_err(|_| Error::malformed(vector.pos()))
    }

    /// The union whose type is field `type_id` and value field `type_id + 1`.
    ///
    /// Returns the type and the value; the value is `None` for type `NONE` (0) or if absent.
    pub(crate) fn union(&self, type_id: u16) -> Result<(u8, Option<Table<'m>>)> {
        let ty: u8 = self.scalar(type_id, 0)?;
        if ty == 0 {
            return Ok((0, None));
        }
        Ok((ty, self.table(type_id + 1)?))
    }
}
