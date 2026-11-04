//! tract: each model is loaded at run time with `tract_tflite`, optimized (`into_optimized`) and
//! made runnable. Each inference builds the int8 input tensor with the model's quantized input
//! type and runs the plan, as tract's API is meant to be used.

use tract_tflite::internal::format_err;
use tract_tflite::prelude::*;

fn main() {
    harness::run(|| {
        for model in harness::MODELS {
            if let Err(e) = bench(model) {
                // The full chain quotes whole flatbuffer tables: keep the operator and the cause.
                let chain = format!("{e:#}");
                let operator = chain
                    .split("builtin_code: ")
                    .nth(1)
                    .and_then(|rest| rest.split([',', ' ']).next())
                    .unwrap_or("?");
                let reason = format!("load fails on {operator}: {}", e.root_cause());
                harness::skip("tract", model, &reason);
                if std::env::var_os("BENCH_DEBUG").is_some() {
                    eprintln!("tract {model}: {chain}");
                }
            }
        }
    });
}

fn bench(model: &str) -> TractResult<()> {
    let path = harness::testdata().join(format!("{model}.tflite"));
    let typed = tract_tflite::tflite().model_for_path(&path)?;
    let input_fact = typed.input_fact(0)?.clone();
    let shape: Vec<usize> = input_fact
        .shape
        .as_concrete()
        .ok_or_else(|| format_err!("symbolic input shape"))?
        .to_vec();
    let datum_type = input_fact.datum_type;
    if std::env::var_os("BENCH_DEBUG").is_some() {
        eprintln!(
            "tract {model}: input fact {input_fact:?}, output fact {:?}",
            typed.output_fact(0)?
        );
    }
    let plan = typed.into_optimized()?.into_runnable()?;

    // Check that one inference works before handing the closure to the harness, which unwraps.
    let run = |input: &[i8], output: &mut [i8]| -> TractResult<()> {
        let mut tensor = Tensor::from_shape(&shape, input)?;
        // SAFETY: the quantized int8 type has the same representation as i8.
        unsafe { tensor.set_datum_type(datum_type) };
        let outputs = plan.run(tvec!(tensor.into()))?;
        let view = outputs[0].to_plain_array_view::<i8>()?;
        for (o, &y) in output.iter_mut().zip(view.iter()) {
            *o = y;
        }
        Ok(())
    };
    let sample = &harness::samples(model)[0];
    run(&sample.input, &mut vec![0; sample.expected.len()])?;

    harness::bench("tract", model, |input, output| run(input, output).unwrap());
    Ok(())
}
