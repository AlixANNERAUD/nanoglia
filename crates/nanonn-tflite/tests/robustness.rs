//! Corrupted models must never make the reader panic.

use nanonn_core::{Detail, Error};
use nanonn_tflite::Model;

fn kws() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/mlperf-tiny/kws_ref_model.tflite"
    );
    std::fs::read(path).unwrap()
}

/// After a successful parse, accessors may report unsupported content but never a malformed model.
fn not_malformed(err: Error) {
    assert!(
        !matches!(err.detail, Detail::MalformedModel { .. }),
        "malformed error after a successful parse: {err}"
    );
}

/// Parses `bytes` and, if it succeeds, reads everything the reader exposes.
fn walk(bytes: &[u8]) {
    let Ok(model) = Model::parse(bytes) else {
        return;
    };
    let _ = model.description().len();
    for s in 0..model.subgraph_count() {
        let graph = match model.subgraph(s) {
            Ok(graph) => graph,
            Err(err) => return not_malformed(err),
        };
        let _ = graph.name().len();
        for t in graph.inputs().iter().chain(graph.outputs().iter()) {
            assert!(t >= 0 && (t as usize) < graph.tensor_count());
        }
        for i in 0..graph.tensor_count() {
            match graph.tensor(i) {
                Ok(t) => {
                    let _ = t.quant.scale.iter().sum::<f32>();
                    let _ = t.quant.zero_point.iter().fold(0i64, i64::wrapping_add);
                    let _ = t.data.map(|d| (d.first(), d.last()));
                    let _ = t.byte_len();
                }
                Err(err) => not_malformed(err),
            }
            if let Err(err) = graph.tensor_is_variable(i) {
                not_malformed(err);
            }
        }
        for i in 0..graph.operator_count() {
            match graph.operator(i) {
                Ok(op) => {
                    for t in op.inputs.iter() {
                        assert!(t == -1 || (t >= 0 && (t as usize) < graph.tensor_count()));
                    }
                    for t in op.outputs.iter() {
                        assert!(t >= 0 && (t as usize) < graph.tensor_count());
                    }
                    let _ = format!("{op:?}");
                }
                Err(err) => not_malformed(err),
            }
        }
    }
}

/// xorshift64*: deterministic, good enough to pick positions and values.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn every_truncation() {
    let bytes = kws();
    for len in 0..bytes.len() {
        walk(&bytes[..len]);
    }
    // The model ends with constant data: truncating it must be detected.
    assert!(Model::parse(&bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn random_corruptions() {
    let original = kws();
    walk(&original);

    // The flatbuffer metadata is much smaller than the weights: aim most corruptions at the
    // bytes that are not constant tensor data.
    let model = Model::parse(&original).unwrap();
    let graph = model.subgraph(0).unwrap();
    let mut is_data = vec![false; original.len()];
    for i in 0..graph.tensor_count() {
        if let Some(data) = graph.tensor(i).unwrap().data {
            let start = data.as_ptr() as usize - original.as_ptr() as usize;
            is_data[start..start + data.len()].fill(true);
        }
    }
    let metadata: Vec<usize> = (0..original.len()).filter(|&i| !is_data[i]).collect();
    assert!(metadata.len() < original.len());

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut bytes = original.clone();
    let mut parsed = 0;
    for iteration in 0..8_000 {
        bytes.copy_from_slice(&original);
        for _ in 0..1 + rng.below(4) {
            let pos = if iteration % 8 == 0 {
                rng.below(bytes.len())
            } else {
                metadata[rng.below(metadata.len())]
            };
            let value = rng.next();
            match rng.below(4) {
                0 => bytes[pos] ^= 1 << (value % 8),
                1 => bytes[pos] = value as u8,
                2 => {
                    // A whole little-endian word: offsets, lengths, indices.
                    let end = (pos + 4).min(bytes.len());
                    let word = match value % 4 {
                        0 => 0,
                        1 => u32::MAX,
                        2 => 0x8000_0000,
                        _ => (value >> 8) as u32 % 4096,
                    };
                    bytes[pos..end].copy_from_slice(&word.to_le_bytes()[..end - pos]);
                }
                _ => {
                    let delta = (value % 7) as u8;
                    bytes[pos] = bytes[pos].wrapping_add(delta).wrapping_sub(3);
                }
            }
        }
        if Model::parse(&bytes).is_ok() {
            parsed += 1;
        }
        walk(&bytes);
    }
    // Many corruptions hit fields that do not matter to the structure.
    assert!(parsed > 0);
}
