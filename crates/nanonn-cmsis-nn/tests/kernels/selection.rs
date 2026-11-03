//! Operators the backend must leave to other backends, and its error paths.

use nanonn_cmsis_nn::BACKEND;
use nanonn_core::backend::{Backend, BackendKind, EvalCtx};
use nanonn_core::{
    Activation, BuiltinOp, Conv2DOptions, DataType, ErrorCode, FullyConnectedOptions, OpOptions,
    Padding,
};

use crate::harness::{Aligned, Op, Tensor};

fn conv_options(activation: Activation) -> OpOptions<'static> {
    OpOptions::Conv2D(Conv2DOptions {
        padding: Padding::Valid,
        stride_w: 1,
        stride_h: 1,
        dilation_w: 1,
        dilation_h: 1,
        activation,
    })
}

#[test]
fn identity() {
    assert_eq!(BACKEND.name(), "cmsis-nn");
    assert_eq!(BACKEND.kind(), BackendKind::Cpu);
}

#[test]
fn conv_conditions() {
    let input = Tensor::activation(DataType::I8, &[1, 4, 4, 2], 0.1, 0);
    let filter = Tensor::weights(&[3, 3, 3, 2], &[1; 54], &[0.01], 0);
    let bias = Tensor::bias(&[0, 0, 0]);
    let output = Tensor::activation(DataType::I8, &[1, 2, 2, 3], 0.2, 0);
    let conv = |activation| {
        Op::new(BuiltinOp::CONV_2D, conv_options(activation))
            .input(&input)
            .input(&filter)
            .input(&bias)
            .output(&output)
            .selected()
    };
    assert!(conv(Activation::Relu));
    // Activations CMSIS-NN cannot fuse.
    assert!(!conv(Activation::Tanh));
    assert!(!conv(Activation::SignBit));

    // Wrong output shape for the options.
    let bad_output = Tensor::activation(DataType::I8, &[1, 3, 3, 3], 0.2, 0);
    assert!(
        !Op::new(BuiltinOp::CONV_2D, conv_options(Activation::None))
            .input(&input)
            .input(&filter)
            .output(&bad_output)
            .selected()
    );
    // Float and uint8 activations.
    let float_input = Tensor::activation(DataType::F32, &[1, 4, 4, 2], 0.1, 0);
    assert!(
        !Op::new(BuiltinOp::CONV_2D, conv_options(Activation::None))
            .input(&float_input)
            .input(&filter)
            .output(&output)
            .selected()
    );
    let u8_input = Tensor::activation(DataType::U8, &[1, 4, 4, 2], 0.1, 128);
    assert!(
        !Op::new(BuiltinOp::CONV_2D, conv_options(Activation::None))
            .input(&u8_input)
            .input(&filter)
            .output(&output)
            .selected()
    );
    // int16 input with an int8 output.
    let i16_input = Tensor::activation(DataType::I16, &[1, 4, 4, 2], 0.1, 0);
    let i16_output = Tensor::activation(DataType::I8, &[1, 2, 2, 3], 0.2, 0);
    assert!(
        !Op::new(BuiltinOp::CONV_2D, conv_options(Activation::None))
            .input(&i16_input)
            .input(&filter)
            .output(&i16_output)
            .selected()
    );
    // Per-channel scales along the wrong axis.
    let wrong_axis = Tensor::weights(&[3, 3, 3, 2], &[1; 54], &[0.01, 0.02, 0.03], 3);
    assert!(
        !Op::new(BuiltinOp::CONV_2D, conv_options(Activation::None))
            .input(&input)
            .input(&wrong_axis)
            .output(&output)
            .selected()
    );
    // A bias of the wrong size.
    let short_bias = Tensor::bias(&[0, 0]);
    assert!(
        !Op::new(BuiltinOp::CONV_2D, conv_options(Activation::None))
            .input(&input)
            .input(&filter)
            .input(&short_bias)
            .output(&output)
            .selected()
    );
}

#[test]
fn fully_connected_conditions() {
    let input = Tensor::activation(DataType::I8, &[1, 4], 0.1, 0);
    let output = Tensor::activation(DataType::I8, &[1, 2], 0.2, 0);
    let options = |default_weights_format| {
        OpOptions::FullyConnected(FullyConnectedOptions {
            activation: Activation::None,
            default_weights_format,
            keep_num_dims: false,
            asymmetric_quantize_inputs: false,
        })
    };
    let filter = Tensor::weights(&[2, 4], &[1; 8], &[0.01], 0);
    let fc = |filter: &Tensor, default_weights_format| {
        Op::new(BuiltinOp::FULLY_CONNECTED, options(default_weights_format))
            .input(&input)
            .input(filter)
            .output(&output)
            .selected()
    };
    assert!(fc(&filter, true));
    assert!(!fc(&filter, false));
    // Asymmetric weights.
    let asymmetric = Tensor::weights(&[2, 4], &[1; 8], &[0.01], 0).with_zero_points(&[3]);
    assert!(!fc(&asymmetric, true));
    // Accumulation depth not matching the input.
    let wide = Tensor::weights(&[2, 3], &[1; 6], &[0.01], 0);
    assert!(!fc(&wide, true));
}

#[test]
fn elementwise_and_softmax_conditions() {
    let a = Tensor::activation(DataType::I8, &[1, 4], 0.1, 0);
    let b = Tensor::activation(DataType::I8, &[1, 4], 0.1, 0);
    let row = Tensor::activation(DataType::I8, &[1, 1], 0.1, 0);
    let out = Tensor::activation(DataType::I8, &[1, 4], 0.2, 0);
    let add = |b: &Tensor| {
        Op::new(
            BuiltinOp::ADD,
            OpOptions::Elementwise {
                activation: Activation::None,
            },
        )
        .input(&a)
        .input(b)
        .output(&out)
        .selected()
    };
    assert!(add(&b));
    // Broadcasting is left to other backends.
    assert!(!add(&row));

    let softmax = |output: &Tensor| {
        Op::new(BuiltinOp::SOFTMAX, OpOptions::Softmax { beta: 1.0 })
            .input(&a)
            .output(output)
            .selected()
    };
    assert!(softmax(&Tensor::activation(
        DataType::I8,
        &[1, 4],
        1.0 / 256.0,
        -128
    )));
    assert!(!softmax(&Tensor::activation(
        DataType::I8,
        &[1, 4],
        1.0 / 256.0,
        0
    )));
    assert!(!softmax(&Tensor::activation(
        DataType::I8,
        &[1, 4],
        1.0 / 128.0,
        -128
    )));

    // Operators the backend does not implement at all.
    assert!(
        !Op::new(BuiltinOp::RESHAPE, OpOptions::None)
            .input(&a)
            .output(&out)
            .selected()
    );
}

#[test]
fn eval_rejects_wrong_buffers() {
    let input = Tensor::activation(DataType::I8, &[1, 4, 4, 2], 0.1, 0);
    let filter = Tensor::weights(&[3, 3, 3, 2], &[1; 54], &[0.01], 0);
    let output = Tensor::activation(DataType::I8, &[1, 2, 2, 3], 0.2, 0);
    let op = Op::new(BuiltinOp::CONV_2D, conv_options(Activation::None))
        .input(&input)
        .input(&filter)
        .output(&output);
    op.with_desc(|desc| {
        let kernel = BACKEND.select(desc).unwrap();
        let plan = (kernel.plan)(desc).unwrap();
        let mut persist = Aligned::zeroed(plan.persistent);
        (kernel.prepare)(desc, persist.bytes_mut()).unwrap();
        let mut scratch = Aligned::zeroed(plan.scratch);
        let short_input = Aligned::zeroed(31);
        let mut out = Aligned::zeroed(12);
        let inputs = [short_input.bytes(), filter.data.bytes()];
        let mut outputs = [out.bytes_mut()];
        let mut ctx = EvalCtx {
            inputs: &inputs,
            outputs: &mut outputs,
            persist: persist.bytes(),
            scratch: scratch.bytes_mut(),
        };
        let error = (kernel.eval)(&mut ctx).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidArgument);

        // A scratch buffer smaller than planned.
        let good_input = Aligned::zeroed(32);
        let inputs = [good_input.bytes(), filter.data.bytes()];
        let mut small = Aligned::zeroed(plan.scratch.saturating_sub(1));
        let mut outputs = [out.bytes_mut()];
        let mut ctx = EvalCtx {
            inputs: &inputs,
            outputs: &mut outputs,
            persist: persist.bytes(),
            scratch: small.bytes_mut(),
        };
        assert!(plan.scratch > 0);
        assert_eq!(
            (kernel.eval)(&mut ctx).unwrap_err().code,
            ErrorCode::InvalidArgument
        );
    });
}
