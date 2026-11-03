//! A tiny flatbuffer writer to build TFLite models field by field in tests.
//!
//! Objects are laid out front to back (a parent before its children), without padding: the
//! reader must not depend on alignment.

#![allow(dead_code)]

/// A flatbuffer object.
#[derive(Clone, Debug)]
pub enum Value {
    /// A table: `(field id, field)` pairs.
    Table(Vec<(u16, Field)>),
    /// A `[ubyte]` vector.
    Bytes(Vec<u8>),
    /// An `[int]` vector.
    Ints(Vec<i32>),
    /// A `[long]` vector.
    Longs(Vec<i64>),
    /// A `[float]` vector.
    Floats(Vec<f32>),
    /// A string (NUL-terminated by the writer).
    Str(Vec<u8>),
    /// A vector of tables.
    Tables(Vec<Value>),
}

/// A table field.
#[derive(Clone, Debug)]
pub enum Field {
    U8(u8),
    I8(i8),
    I32(i32),
    U32(u32),
    U64(u64),
    F32(f32),
    /// An offset to another object.
    Ref(Value),
}

impl Field {
    fn inline(&self) -> Vec<u8> {
        match self {
            Field::U8(v) => v.to_le_bytes().to_vec(),
            Field::I8(v) => v.to_le_bytes().to_vec(),
            Field::I32(v) => v.to_le_bytes().to_vec(),
            Field::U32(v) => v.to_le_bytes().to_vec(),
            Field::U64(v) => v.to_le_bytes().to_vec(),
            Field::F32(v) => v.to_le_bytes().to_vec(),
            Field::Ref(_) => vec![0; 4],
        }
    }
}

/// Shorthand for a string field.
pub fn s(text: &str) -> Field {
    Field::Ref(Value::Str(text.as_bytes().to_vec()))
}

/// Shorthand for an `[int]` field.
pub fn ints(values: &[i32]) -> Field {
    Field::Ref(Value::Ints(values.to_vec()))
}

/// Shorthand for a table field.
pub fn table(fields: Vec<(u16, Field)>) -> Field {
    Field::Ref(Value::Table(fields))
}

/// Shorthand for a vector of tables field.
pub fn tables(values: Vec<Vec<(u16, Field)>>) -> Field {
    Field::Ref(Value::Tables(
        values.into_iter().map(Value::Table).collect(),
    ))
}

struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn patch(&mut self, slot: usize, target: usize) {
        let offset = u32::try_from(target - slot).unwrap();
        self.out[slot..slot + 4].copy_from_slice(&offset.to_le_bytes());
    }

    fn vector(&mut self, len: usize, bytes: &[u8]) -> usize {
        let pos = self.out.len();
        self.out.extend_from_slice(&(len as u32).to_le_bytes());
        self.out.extend_from_slice(bytes);
        pos
    }

    fn write(&mut self, value: &Value) -> usize {
        match value {
            Value::Table(fields) => {
                let slots = fields.iter().map(|(id, _)| *id + 1).max().unwrap_or(0);
                let vtable_len = 4 + 2 * usize::from(slots);
                let inline_len = 4 + fields.iter().map(|(_, f)| f.inline().len()).sum::<usize>();
                let vtable = self.out.len();
                let table = vtable + vtable_len;
                let mut voffsets = vec![0u16; usize::from(slots)];
                let mut body = Vec::new();
                let mut refs = Vec::new();
                for (id, field) in fields {
                    let voffset = 4 + body.len();
                    voffsets[usize::from(*id)] = voffset as u16;
                    if let Field::Ref(child) = field {
                        refs.push((table + voffset, child));
                    }
                    body.extend(field.inline());
                }
                self.out.extend((vtable_len as u16).to_le_bytes());
                self.out.extend((inline_len as u16).to_le_bytes());
                for v in voffsets {
                    self.out.extend(v.to_le_bytes());
                }
                self.out.extend(((table - vtable) as i32).to_le_bytes());
                self.out.extend(body);
                for (slot, child) in refs {
                    let target = self.write(child);
                    self.patch(slot, target);
                }
                table
            }
            Value::Bytes(v) => self.vector(v.len(), v),
            Value::Ints(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                self.vector(v.len(), &bytes)
            }
            Value::Longs(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                self.vector(v.len(), &bytes)
            }
            Value::Floats(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                self.vector(v.len(), &bytes)
            }
            Value::Str(v) => {
                let pos = self.vector(v.len(), v);
                self.out.push(0);
                pos
            }
            Value::Tables(v) => {
                let pos = self.vector(v.len(), &vec![0; 4 * v.len()]);
                for (i, child) in v.iter().enumerate() {
                    let target = self.write(child);
                    self.patch(pos + 4 + 4 * i, target);
                }
                pos
            }
        }
    }
}

/// Serializes `root` as a TFLite file (with the `TFL3` identifier).
pub fn finish(root: &Value) -> Vec<u8> {
    let mut w = Writer {
        out: b"\0\0\0\0TFL3".to_vec(),
    };
    let pos = w.write(root);
    w.patch(0, pos);
    w.out
}

/// Field ids used by the tests (see the TFLite schema).
pub mod id {
    pub mod model {
        pub const VERSION: u16 = 0;
        pub const OPERATOR_CODES: u16 = 1;
        pub const SUBGRAPHS: u16 = 2;
        pub const DESCRIPTION: u16 = 3;
        pub const BUFFERS: u16 = 4;
    }
    pub mod opcode {
        pub const DEPRECATED_BUILTIN_CODE: u16 = 0;
        pub const CUSTOM_CODE: u16 = 1;
        pub const VERSION: u16 = 2;
        pub const BUILTIN_CODE: u16 = 3;
    }
    pub mod subgraph {
        pub const TENSORS: u16 = 0;
        pub const INPUTS: u16 = 1;
        pub const OUTPUTS: u16 = 2;
        pub const OPERATORS: u16 = 3;
        pub const NAME: u16 = 4;
    }
    pub mod tensor {
        pub const SHAPE: u16 = 0;
        pub const TYPE: u16 = 1;
        pub const BUFFER: u16 = 2;
        pub const NAME: u16 = 3;
        pub const QUANTIZATION: u16 = 4;
        pub const IS_VARIABLE: u16 = 5;
        pub const SPARSITY: u16 = 6;
        pub const SHAPE_SIGNATURE: u16 = 7;
    }
    pub mod quant {
        pub const SCALE: u16 = 2;
        pub const ZERO_POINT: u16 = 3;
        pub const DETAILS_TYPE: u16 = 4;
        pub const DETAILS: u16 = 5;
        pub const QUANTIZED_DIMENSION: u16 = 6;
    }
    pub mod operator {
        pub const OPCODE_INDEX: u16 = 0;
        pub const INPUTS: u16 = 1;
        pub const OUTPUTS: u16 = 2;
        pub const BUILTIN_OPTIONS_TYPE: u16 = 3;
        pub const BUILTIN_OPTIONS: u16 = 4;
        pub const CUSTOM_OPTIONS: u16 = 5;
        pub const LARGE_CUSTOM_OPTIONS_OFFSET: u16 = 9;
        pub const LARGE_CUSTOM_OPTIONS_SIZE: u16 = 10;
    }
    pub mod buffer {
        pub const DATA: u16 = 0;
        pub const OFFSET: u16 = 1;
        pub const SIZE: u16 = 2;
    }
}
