//! Up-front verification of a whole TFLite model.
//!
//! Walks every table reachable from the root, following the schema: offsets, vtables, vectors
//! and strings must be in bounds, strings must be NUL-terminated UTF-8, and the indices nanoglia
//! follows (operator codes, buffers, tensors) must be in range. The schema has no recursive
//! table, so the nesting depth is bounded by construction. A work budget proportional to the
//! model size bounds the time spent on adversarial models that share structures many times.

use nanoglia_core::le::LeSlice;
use nanoglia_core::{Error, OpCode, Result};

use crate::fb::{Table, Tables};
use crate::options;
use crate::schema::{
    buffer, dimension_metadata, external_buffer, external_buffer_group, metadata, model, operator,
    operator_code, quantization, signature_def, sparse_index_vector, sparsity, subgraph, tensor,
    tensor_map, variant_sub_type,
};

/// Bounds the work done on a model.
struct Budget(usize);

impl Budget {
    /// Spends `units`, failing with a malformed model at `pos` when the budget is exhausted.
    fn spend(&mut self, units: usize, pos: usize) -> Result<()> {
        self.0 = self.0.checked_sub(units).ok_or(Error::malformed(pos))?;
        Ok(())
    }
}

/// Verifier state.
struct Verifier {
    budget: Budget,
}

impl Verifier {
    /// Accounts for a table visit.
    fn visit(&mut self, table: &Table<'_>) -> Result<()> {
        self.budget.spend(4, table.pos())
    }

    /// Visits each table of `tables` with `f`.
    fn each<'m>(
        &mut self,
        tables: Tables<'m>,
        mut f: impl FnMut(&mut Self, Table<'m>) -> Result<()>,
    ) -> Result<()> {
        for i in 0..tables.len() {
            let table = tables.get(i)?;
            self.visit(&table)?;
            f(self, table)?;
        }
        Ok(())
    }

    /// Checks the string field `id`, accounting for its UTF-8 validation.
    fn str(&mut self, table: &Table<'_>, id: u16) -> Result<()> {
        if let Some(s) = table.str(id)? {
            self.budget.spend(s.len(), table.pos())?;
        }
        Ok(())
    }

    /// Checks that every index of `indices` is in `0..count`, or `-1` if `optional`.
    fn indices(
        &mut self,
        indices: LeSlice<'_, i32>,
        count: usize,
        optional: bool,
        pos: usize,
    ) -> Result<()> {
        self.budget.spend(indices.len().saturating_mul(4), pos)?;
        let valid = |i: i32| match usize::try_from(i) {
            Ok(i) => i < count,
            Err(_) => optional && i == -1,
        };
        if indices.iter().all(valid) {
            Ok(())
        } else {
            Err(Error::malformed(pos))
        }
    }

    /// Checks a table nanoglia never reads (an unknown union member): only its vtable and bounds.
    fn opaque(&mut self, table: Option<Table<'_>>) -> Result<()> {
        match table {
            Some(table) => self.visit(&table),
            None => Ok(()),
        }
    }

    fn model(&mut self, root: &Table<'_>) -> Result<()> {
        self.visit(root)?;
        root.scalar::<u32>(model::VERSION, 0)?;
        self.str(root, model::DESCRIPTION)?;

        let buffers = root.tables(model::BUFFERS)?;
        self.each(buffers, |_, b| {
            b.bytes(buffer::DATA)?;
            let offset: u64 = b.scalar(buffer::OFFSET, 0)?;
            let size: u64 = b.scalar(buffer::SIZE, 0)?;
            if offset > 1 && crate::external(b.buf(), offset, size).is_none() {
                return Err(Error::malformed(b.pos()));
            }
            Ok(())
        })?;

        let opcodes = root.tables(model::OPERATOR_CODES)?;
        self.each(opcodes, |v, code| {
            code.scalar::<i8>(operator_code::DEPRECATED_BUILTIN_CODE, 0)?;
            v.str(&code, operator_code::CUSTOM_CODE)?;
            code.scalar::<i32>(operator_code::VERSION, 1)?;
            code.scalar::<i32>(operator_code::BUILTIN_CODE, 0)?;
            crate::op_code(&code)?;
            Ok(())
        })?;

        let counts = Counts {
            buffers: buffers.len(),
            opcodes,
        };
        self.each(root.tables(model::SUBGRAPHS)?, |v, sg| {
            v.subgraph(&sg, &counts)
        })?;

        root.slice::<i32>(model::METADATA_BUFFER)?;
        self.each(root.tables(model::METADATA)?, |v, m| {
            v.str(&m, metadata::NAME)?;
            m.scalar::<u32>(metadata::BUFFER, 0)?;
            Ok(())
        })?;
        self.each(root.tables(model::SIGNATURE_DEFS)?, |v, s| {
            for id in [signature_def::INPUTS, signature_def::OUTPUTS] {
                v.each(s.tables(id)?, |v, m| {
                    v.str(&m, tensor_map::NAME)?;
                    m.scalar::<u32>(tensor_map::TENSOR_INDEX, 0)?;
                    Ok(())
                })?;
            }
            v.str(&s, signature_def::SIGNATURE_KEY)?;
            v.str(&s, signature_def::DEPRECATED_TAG)?;
            s.scalar::<u32>(signature_def::SUBGRAPH_INDEX, 0)?;
            Ok(())
        })?;
        self.each(root.tables(model::EXTERNAL_BUFFER_GROUPS)?, |v, g| {
            v.str(&g, external_buffer_group::NAME)
        })?;
        self.each(root.tables(model::EXTERNAL_BUFFERS)?, |v, b| {
            b.scalar::<u32>(external_buffer::ID, 0)?;
            b.scalar::<u32>(external_buffer::GROUP, 0)?;
            b.scalar::<u64>(external_buffer::OFFSET, 0)?;
            b.scalar::<u64>(external_buffer::LENGTH, 0)?;
            v.str(&b, external_buffer::PACKING)
        })?;
        Ok(())
    }

    fn subgraph(&mut self, sg: &Table<'_>, counts: &Counts<'_>) -> Result<()> {
        let tensors = sg.tables(subgraph::TENSORS)?;
        self.each(tensors, |v, t| v.tensor(&t, counts))?;
        let tensor_count = tensors.len();
        for id in [subgraph::INPUTS, subgraph::OUTPUTS] {
            self.indices(sg.slice(id)?, tensor_count, false, sg.pos())?;
        }
        self.each(sg.tables(subgraph::OPERATORS)?, |v, op| {
            v.operator(&op, counts, tensor_count)
        })?;
        self.str(sg, subgraph::NAME)?;
        sg.scalar::<i32>(subgraph::DEBUG_METADATA_INDEX, -1)?;
        Ok(())
    }

    fn tensor(&mut self, t: &Table<'_>, counts: &Counts<'_>) -> Result<()> {
        t.slice::<i32>(tensor::SHAPE)?;
        t.scalar::<i8>(tensor::TYPE, 0)?;
        let buffer: u32 = t.scalar(tensor::BUFFER, 0)?;
        if buffer as usize >= counts.buffers.max(1) {
            // Buffer 0 is the empty sentinel, valid even when the buffer list is absent.
            return Err(Error::malformed(t.pos()));
        }
        self.str(t, tensor::NAME)?;
        if let Some(q) = t.table(tensor::QUANTIZATION)? {
            self.visit(&q)?;
            for id in [quantization::MIN, quantization::MAX, quantization::SCALE] {
                q.slice::<f32>(id)?;
            }
            q.slice::<i64>(quantization::ZERO_POINT)?;
            self.opaque(q.union(quantization::DETAILS_TYPE)?.1)?;
            q.scalar::<i32>(quantization::QUANTIZED_DIMENSION, 0)?;
        }
        t.flag(tensor::IS_VARIABLE, false)?;
        if let Some(s) = t.table(tensor::SPARSITY)? {
            self.visit(&s)?;
            s.slice::<i32>(sparsity::TRAVERSAL_ORDER)?;
            s.slice::<i32>(sparsity::BLOCK_MAP)?;
            self.each(s.tables(sparsity::DIM_METADATA)?, |v, d| {
                d.scalar::<i8>(dimension_metadata::FORMAT, 0)?;
                d.scalar::<i32>(dimension_metadata::DENSE_SIZE, 0)?;
                for id in [
                    dimension_metadata::ARRAY_SEGMENTS_TYPE,
                    dimension_metadata::ARRAY_INDICES_TYPE,
                ] {
                    v.sparse_index_vector(&d, id)?;
                }
                Ok(())
            })?;
        }
        t.slice::<i32>(tensor::SHAPE_SIGNATURE)?;
        t.flag(tensor::HAS_RANK, false)?;
        self.each(t.tables(tensor::VARIANT_TENSORS)?, |_, s| {
            s.slice::<i32>(variant_sub_type::SHAPE)?;
            s.scalar::<i8>(variant_sub_type::TYPE, 0)?;
            s.flag(variant_sub_type::HAS_RANK, false)?;
            Ok(())
        })?;
        t.scalar::<u32>(tensor::EXTERNAL_BUFFER, 0)?;
        Ok(())
    }

    fn sparse_index_vector(&mut self, d: &Table<'_>, type_id: u16) -> Result<()> {
        let (ty, value) = d.union(type_id)?;
        let Some(value) = value else {
            return Ok(());
        };
        self.visit(&value)?;
        let elem_size = match ty {
            sparse_index_vector::INT32 => 4,
            sparse_index_vector::UINT16 => 2,
            sparse_index_vector::UINT8 => 1,
            _ => return Ok(()),
        };
        value.vector(sparse_index_vector::VALUES, elem_size)?;
        Ok(())
    }

    fn operator(&mut self, op: &Table<'_>, counts: &Counts<'_>, tensor_count: usize) -> Result<()> {
        let opcode_index: u32 = op.scalar(operator::OPCODE_INDEX, 0)?;
        let code = counts
            .opcodes
            .get(opcode_index as usize)
            .map_err(|_| Error::malformed(op.pos()))?;
        self.indices(op.slice(operator::INPUTS)?, tensor_count, true, op.pos())?;
        self.indices(op.slice(operator::OUTPUTS)?, tensor_count, false, op.pos())?;

        let (ty, value) = op.union(operator::BUILTIN_OPTIONS_TYPE)?;
        if let Some(value) = &value {
            self.visit(value)?;
            options::verify(ty, value)?;
        }
        // The options must be those of the operator, so that `options::decode` cannot fail on
        // the structure later.
        if ty != 0
            && let OpCode::Builtin(builtin) = crate::op_code(&code)?
            && let Some(expected) = options::expected_type(builtin)
            && ty != expected
        {
            return Err(Error::malformed(op.pos()));
        }

        op.bytes(operator::CUSTOM_OPTIONS)?;
        op.scalar::<i8>(operator::CUSTOM_OPTIONS_FORMAT, 0)?;
        op.bytes(operator::MUTATING_VARIABLE_INPUTS)?;
        op.slice::<i32>(operator::INTERMEDIATES)?;
        crate::large_custom_options(op)?;
        self.opaque(op.union(operator::BUILTIN_OPTIONS_2_TYPE)?.1)?;
        op.scalar::<i32>(operator::DEBUG_METADATA_INDEX, -1)?;
        Ok(())
    }
}

/// Model-wide values the per-table checks need.
struct Counts<'m> {
    buffers: usize,
    opcodes: Tables<'m>,
}

/// Verifies the whole model whose root table is `root`.
pub(crate) fn model(root: &Table<'_>) -> Result<()> {
    // Every unshared structure costs at most its size in bytes, so honest models stay well
    // within this budget while shared-structure bombs are cut short.
    let budget = root.buf().len().saturating_mul(2).saturating_add(4096);
    Verifier {
        budget: Budget(budget),
    }
    .model(root)
}
