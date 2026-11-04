//! The C API called from Rust: memory sizing, loading, the golden keyword-spotting samples with
//! `compute` and `step`, cancellation, and every invalid argument.

use core::ffi::{CStr, c_void};
use core::ptr;
use nanoglia_c::*;
use std::path::{Path, PathBuf};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

/// Bytes on a 16-byte boundary, at `offset` bytes past it.
struct Buffer {
    storage: Vec<u128>,
    offset: usize,
    len: usize,
}

impl Buffer {
    fn zeroed(len: usize, offset: usize) -> Self {
        Self {
            storage: vec![0; (len + offset).div_ceil(16)],
            offset,
            len,
        }
    }

    fn from(bytes: &[u8]) -> Self {
        let mut buffer = Self::zeroed(bytes.len(), 0);
        buffer.bytes_mut().copy_from_slice(bytes);
        buffer
    }

    fn ptr(&mut self) -> *mut u8 {
        // SAFETY: `offset` is within the storage.
        unsafe { self.storage.as_mut_ptr().cast::<u8>().add(self.offset) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        let len = self.len;
        // SAFETY: `len` bytes from `offset` are within the storage.
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), len) }
    }
}

fn kws() -> Buffer {
    Buffer::from(&std::fs::read(testdata().join("mlperf-tiny/kws_ref_model.tflite")).unwrap())
}

fn sample(k: usize, name: &str) -> Vec<u8> {
    std::fs::read(testdata().join(format!(
        "mlperf-tiny/golden/kws_ref_model/sample_{k}/{name}"
    )))
    .unwrap()
}

fn error() -> ErrorInfo {
    ErrorInfo {
        status: Status::Unknown,
        detail: DetailKind::Unknown,
        index: u32::MAX,
        code: -1,
        size: usize::MAX,
        actual: usize::MAX,
        backend: [1; BACKEND_NAME_SIZE],
    }
}

fn requirements(model: &mut Buffer) -> Memory {
    let mut scratch = vec![0u8; 64 * 1024];
    let mut memory = Memory {
        graph_bytes: 0,
        context_bytes: 0,
        memory_bytes: 0,
    };
    let mut report = error();
    // SAFETY: valid pointers and lengths.
    let status = unsafe {
        nanoglia_memory_requirements(
            model.ptr(),
            model.len,
            scratch.as_mut_ptr().cast(),
            scratch.len(),
            &mut memory,
            &mut report,
        )
    };
    assert_eq!(status, Status::Ok);
    assert_eq!(report.status, Status::Ok);
    assert_eq!(report.detail, DetailKind::None);
    memory
}

fn load(model: &mut Buffer, memory: &mut Buffer) -> Result<*mut Nanoglia, ErrorInfo> {
    let mut nn = ptr::null_mut();
    let mut report = error();
    // SAFETY: valid pointers and lengths; the buffers outlive the handle in every test.
    let status = unsafe {
        nanoglia_load(
            model.ptr(),
            model.len,
            memory.ptr().cast(),
            memory.len,
            &mut nn,
            &mut report,
        )
    };
    assert_eq!(status, report.status);
    if status == Status::Ok {
        assert!(!nn.is_null());
        Ok(nn)
    } else {
        assert!(nn.is_null());
        Err(report)
    }
}

/// Copies golden sample `k`'s input into the handle.
fn write_input(nn: *mut Nanoglia, k: usize) {
    let input = sample(k, "input_0.bin");
    let mut tensor = core::mem::MaybeUninit::<Tensor>::uninit();
    // SAFETY: a live handle and a writable tensor.
    let status = unsafe { nanoglia_input(nn, 0, tensor.as_mut_ptr()) };
    assert_eq!(status, Status::Ok);
    // SAFETY: written on success.
    let tensor = unsafe { tensor.assume_init() };
    assert_eq!(tensor.size, input.len());
    // SAFETY: `size` writable bytes in the handle's buffer.
    unsafe { ptr::copy_nonoverlapping(input.as_ptr(), tensor.data.cast(), input.len()) };
}

fn output(nn: *const Nanoglia) -> Vec<u8> {
    let mut tensor = core::mem::MaybeUninit::<Tensor>::uninit();
    // SAFETY: a live handle and a writable tensor.
    let status = unsafe { nanoglia_output(nn, 0, tensor.as_mut_ptr()) };
    assert_eq!(status, Status::Ok);
    // SAFETY: written on success; `size` readable bytes.
    unsafe {
        let tensor = tensor.assume_init();
        std::slice::from_raw_parts(tensor.data.cast::<u8>(), tensor.size).to_vec()
    }
}

fn release(nn: *mut Nanoglia) {
    // SAFETY: a live handle.
    assert_eq!(unsafe { nanoglia_release(nn) }, Status::Ok);
}

#[test]
fn golden_samples_with_compute_and_step() {
    let mut model = kws();
    let needs = requirements(&mut model);
    let mut memory = Buffer::zeroed(needs.memory_bytes, 0);
    let nn = load(&mut model, &mut memory).unwrap();
    // SAFETY: a live handle.
    let operators = unsafe { nanoglia_operator_count(nn) };
    assert!(operators > 1);

    for k in 0..3 {
        let expected = sample(k, "output_0.bin");
        write_input(nn, k);
        let mut report = error();
        // SAFETY: a live handle.
        assert_eq!(unsafe { nanoglia_compute(nn, &mut report) }, Status::Ok);
        assert_eq!(report.status, Status::Ok);
        assert_eq!(output(nn), expected, "sample {k} with compute");

        write_input(nn, k);
        let mut steps = 0;
        let mut done = false;
        while !done {
            // SAFETY: a live handle.
            let status = unsafe { nanoglia_step(nn, &mut done, ptr::null_mut()) };
            assert_eq!(status, Status::Ok);
            steps += 1;
        }
        assert_eq!(steps, operators);
        assert_eq!(output(nn), expected, "sample {k} with step");
    }
    release(nn);
}

#[test]
fn cancel_starts_the_inference_over() {
    let mut model = kws();
    let needs = requirements(&mut model);
    let mut memory = Buffer::zeroed(needs.memory_bytes, 0);
    let nn = load(&mut model, &mut memory).unwrap();
    let expected = sample(1, "output_0.bin");

    write_input(nn, 1);
    let mut done = false;
    // SAFETY: a live handle.
    unsafe {
        for _ in 0..3 {
            assert_eq!(nanoglia_step(nn, &mut done, ptr::null_mut()), Status::Ok);
            assert!(!done);
        }
        assert_eq!(nanoglia_cancel(nn), Status::Ok);
    }
    // The first operators overwrote memory the input shared: write it again.
    write_input(nn, 1);
    let mut steps = 0;
    while !done {
        // SAFETY: a live handle.
        let status = unsafe { nanoglia_step(nn, &mut done, ptr::null_mut()) };
        assert_eq!(status, Status::Ok);
        steps += 1;
    }
    // SAFETY: a live handle.
    assert_eq!(steps, unsafe { nanoglia_operator_count(nn) });
    assert_eq!(output(nn), expected);
    release(nn);
}

#[test]
fn memory_bytes_is_exact_at_any_alignment() {
    let mut model = kws();
    let needs = requirements(&mut model);
    assert!(needs.memory_bytes > needs.graph_bytes + needs.context_bytes);
    for offset in 0..16 {
        let mut memory = Buffer::zeroed(needs.memory_bytes, offset);
        let nn = load(&mut model, &mut memory).unwrap();
        write_input(nn, 0);
        // SAFETY: a live handle.
        assert_eq!(unsafe { nanoglia_compute(nn, ptr::null_mut()) }, Status::Ok);
        assert_eq!(output(nn), sample(0, "output_0.bin"));
        release(nn);
    }
    // One byte less does not fit when the buffer starts one byte past a 16-byte boundary.
    let mut memory = Buffer::zeroed(needs.memory_bytes - 1, 1);
    let report = load(&mut model, &mut memory).unwrap_err();
    assert_eq!(report.status, Status::TooLarge);
    assert_eq!(report.detail, DetailKind::OutOfMemory);
}

#[test]
fn small_buffers_are_too_large_errors() {
    let mut model = kws();
    let needs = requirements(&mut model);
    for len in [0, 1, 64, needs.memory_bytes / 2] {
        let mut memory = Buffer::zeroed(len, 0);
        let report = load(&mut model, &mut memory).unwrap_err();
        assert_eq!(report.status, Status::TooLarge, "{len} bytes");
        assert_eq!(report.detail, DetailKind::OutOfMemory);
        assert!(report.size > 0);
    }

    let mut scratch = [0u8; 8];
    let mut memory = core::mem::MaybeUninit::<Memory>::uninit();
    let mut report = error();
    // SAFETY: valid pointers and lengths.
    let status = unsafe {
        nanoglia_memory_requirements(
            model.ptr(),
            model.len,
            scratch.as_mut_ptr().cast(),
            scratch.len(),
            memory.as_mut_ptr(),
            &mut report,
        )
    };
    assert_eq!(status, Status::TooLarge);
    assert_eq!(report.detail, DetailKind::OutOfMemory);
}

#[test]
fn corrupted_models_are_errors() {
    let original = std::fs::read(testdata().join("mlperf-tiny/kws_ref_model.tflite")).unwrap();
    let mut memory = Buffer::zeroed(256 * 1024, 0);

    let mut garbage = Buffer::from(b"not a model, not even close");
    let report = load(&mut garbage, &mut memory).unwrap_err();
    assert_eq!(report.status, Status::InvalidEncoding);

    let mut truncated = Buffer::from(&original[..original.len() / 2]);
    let report = load(&mut truncated, &mut memory).unwrap_err();
    assert_eq!(report.status, Status::InvalidEncoding);
    assert_eq!(report.detail, DetailKind::MalformedModel);

    // Flip bytes all over the model: each load is an error or a model that runs, never a crash.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..200 {
        let mut corrupted = Buffer::from(&original);
        for _ in 0..8 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let at = (state as usize) % original.len();
            corrupted.bytes_mut()[at] ^= (state >> 32) as u8 | 1;
        }
        if let Ok(nn) = load(&mut corrupted, &mut memory) {
            // SAFETY: a live handle.
            unsafe { nanoglia_compute(nn, ptr::null_mut()) };
            release(nn);
        }
    }
}

#[test]
fn tensors_and_lookups() {
    let mut model = kws();
    let needs = requirements(&mut model);
    let mut memory = Buffer::zeroed(needs.memory_bytes, 0);
    let nn = load(&mut model, &mut memory).unwrap();
    // SAFETY: a live handle, valid pointers.
    unsafe {
        assert_eq!(nanoglia_input_count(nn), 1);
        assert_eq!(nanoglia_output_count(nn), 1);

        let mut tensor = core::mem::MaybeUninit::<Tensor>::uninit();
        assert_eq!(nanoglia_input(nn, 0, tensor.as_mut_ptr()), Status::Ok);
        let input = tensor.assume_init();
        assert_eq!(input.element_type, ElementType::Int8);
        assert_eq!(input.rank, 4);
        assert_eq!(input.dims, [1, 49, 10, 1, 0, 0]);
        assert_eq!(input.size, 490);
        assert_eq!(input.scale, 0.584_702_9);
        assert_eq!(input.zero_point, 83);
        let name = std::slice::from_raw_parts(input.name.cast::<u8>(), input.name_length);
        assert_eq!(name, b"input_1");

        assert_eq!(nanoglia_output(nn, 0, tensor.as_mut_ptr()), Status::Ok);
        let output = tensor.assume_init();
        assert_eq!(output.element_type, ElementType::Int8);
        assert_eq!(&output.dims[..2], &[1, 12]);
        assert_eq!(output.size, 12);
        assert_eq!(output.scale, 0.003_906_25);
        assert_eq!(output.zero_point, -128);

        assert_eq!(
            nanoglia_input(nn, 1, tensor.as_mut_ptr()),
            Status::InvalidArgument
        );
        assert_eq!(
            nanoglia_output(nn, 1, tensor.as_mut_ptr()),
            Status::InvalidArgument
        );

        let mut index = usize::MAX;
        assert_eq!(
            nanoglia_input_index(nn, c"input_1".as_ptr(), 7, &mut index),
            Status::Ok
        );
        assert_eq!(index, 0);
        assert_eq!(
            nanoglia_output_index(nn, c"Identity".as_ptr(), 8, &mut index),
            Status::Ok
        );
        assert_eq!(index, 0);
        assert_eq!(
            nanoglia_output_index(nn, c"Identity".as_ptr(), 3, &mut index),
            Status::NotFound
        );
        assert_eq!(
            nanoglia_input_index(nn, b"\xff".as_ptr().cast(), 1, &mut index),
            Status::NotFound
        );
    }
    release(nn);
}

#[test]
fn invalid_arguments() {
    let mut model = kws();
    let needs = requirements(&mut model);
    let mut memory = Buffer::zeroed(needs.memory_bytes, 0);
    let mut scratch = [0u8; 1024];
    let mut sizes = core::mem::MaybeUninit::<Memory>::uninit();
    let mut nn = ptr::null_mut();
    let mut report = error();
    let null: *mut Nanoglia = ptr::null_mut();
    let mut tensor = core::mem::MaybeUninit::<Tensor>::uninit();
    let mut index = 0usize;
    let mut done = false;
    // SAFETY: null pointers, or valid ones.
    unsafe {
        let m = model.ptr();
        let s = scratch.as_mut_ptr().cast::<c_void>();
        let n = model.len;
        for status in [
            nanoglia_memory_requirements(ptr::null(), n, s, 1024, sizes.as_mut_ptr(), &mut report),
            nanoglia_memory_requirements(
                m,
                n,
                ptr::null_mut(),
                1024,
                sizes.as_mut_ptr(),
                &mut report,
            ),
            nanoglia_memory_requirements(m, n, s, 1024, ptr::null_mut(), &mut report),
            // The model one byte past a 16-byte boundary.
            nanoglia_memory_requirements(m.add(1), n - 1, s, 1024, sizes.as_mut_ptr(), &mut report),
            nanoglia_load(
                ptr::null(),
                n,
                memory.ptr().cast(),
                memory.len,
                &mut nn,
                &mut report,
            ),
            nanoglia_load(m, n, ptr::null_mut(), memory.len, &mut nn, &mut report),
            nanoglia_load(
                m,
                n,
                memory.ptr().cast(),
                memory.len,
                ptr::null_mut(),
                &mut report,
            ),
            nanoglia_load(
                m.add(1),
                n - 1,
                memory.ptr().cast(),
                memory.len,
                &mut nn,
                &mut report,
            ),
            nanoglia_release(null),
            nanoglia_input(null, 0, tensor.as_mut_ptr()),
            nanoglia_output(null, 0, tensor.as_mut_ptr()),
            nanoglia_input_index(null, c"input_1".as_ptr(), 7, &mut index),
            nanoglia_output_index(null, c"Identity".as_ptr(), 8, &mut index),
            nanoglia_compute(null, &mut report),
            nanoglia_step(null, &mut done, &mut report),
            nanoglia_cancel(null),
        ] {
            assert_eq!(status, Status::InvalidArgument);
        }
        assert_eq!(report.status, Status::InvalidArgument);
        assert_eq!(report.detail, DetailKind::None);
        assert!(nn.is_null());
        assert_eq!(nanoglia_input_count(null), 0);
        assert_eq!(nanoglia_output_count(null), 0);
        assert_eq!(nanoglia_operator_count(null), 0);

        // Null out-parameters of a live handle.
        let nn = load(&mut model, &mut memory).unwrap();
        assert_eq!(
            nanoglia_input(nn, 0, ptr::null_mut()),
            Status::InvalidArgument
        );
        assert_eq!(
            nanoglia_output(nn, 0, ptr::null_mut()),
            Status::InvalidArgument
        );
        assert_eq!(
            nanoglia_input_index(nn, ptr::null(), 0, &mut index),
            Status::InvalidArgument
        );
        assert_eq!(
            nanoglia_input_index(nn, c"input_1".as_ptr(), 7, ptr::null_mut()),
            Status::InvalidArgument
        );
        assert_eq!(
            nanoglia_step(nn, ptr::null_mut(), &mut report),
            Status::InvalidArgument
        );

        // A released handle is no longer one, as long as its buffer is not reused.
        release(nn);
        assert_eq!(nanoglia_release(nn), Status::InvalidArgument);
        assert_eq!(nanoglia_compute(nn, &mut report), Status::InvalidArgument);
        assert_eq!(nanoglia_input_count(nn), 0);
        // Memory that never held a handle is not one either.
        let mut zeros = Buffer::zeroed(4096, 0);
        assert_eq!(
            nanoglia_compute(zeros.ptr().cast(), &mut report),
            Status::InvalidArgument
        );
    }
}

#[test]
fn descriptions() {
    let text = |p| {
        // SAFETY: the descriptions are static NUL-terminated strings.
        unsafe { CStr::from_ptr(p) }.to_str().unwrap()
    };
    assert_eq!(
        text(nanoglia_status_description(Status::Ok as i32)),
        "success"
    );
    assert_eq!(
        text(nanoglia_status_description(Status::TooLarge as i32)),
        "too large"
    );
    assert_eq!(text(nanoglia_status_description(42)), "unknown status");
    assert_eq!(text(nanoglia_status_description(-1)), "unknown status");
    assert_eq!(
        text(nanoglia_detail_description(DetailKind::OutOfMemory as i32)),
        "out of memory"
    );
    assert_eq!(text(nanoglia_detail_description(99)), "unknown detail");
}
