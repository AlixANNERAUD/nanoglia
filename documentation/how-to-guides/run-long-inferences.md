# Run long inferences without blocking

On a microcontroller, one inference of a vision model can take tens or hundreds of
milliseconds. `compute()` runs every operator before returning, which is too long for a
firmware that must keep a radio link alive, refresh a display or serve other tasks. An
execution context can also run an inference **one operator at a time**, or as a **future** that
yields to the async executor between operators, and an observer can stop it at a deadline.

The granularity is one operator: nanoglia never interrupts an operator in the middle (a large
convolution runs to its end). The [Performance](../reference/performance.md) page and an
[observer](write-a-backend.md) timing each operator tell you how long the longest one takes on
your model.

## Step by step, from any main loop

`ExecutionContext::step` runs the next operator of the current inference, starting a new
inference on the current inputs if none is in progress, and returns `Step::Pending` until the
last operator, then `Step::Done`:

```rust,ignore
use nanoglia::Step;

ctx.input_mut(0)?.as_mut_slice::<i8>()?.copy_from_slice(&features);
loop {
    match ctx.step()? {
        Step::Pending => {
            radio.poll();      // anything else the firmware must do between two operators
            display.refresh();
        }
        Step::Done => break,
    }
}
let scores = ctx.output(0)?.as_slice::<i8>()?;
```

- Stop at any time: stop calling `step`, or call `ctx.cancel()`. The next `step` starts a new
  inference. `ctx.next_operator()` tells whether one is in progress, and how far.
- Do not change the inputs while an inference is in progress, and read the outputs only after
  `Step::Done`: in between they hold partial results.
- If an operator fails, `step` returns its error and abandons the inference.

`step` needs no executor, no interrupt and no allocation: it works in a bare `loop`, a
super-loop or an RTOS task.

## As a future, with an async executor

`compute_async` runs the whole inference and yields to the executor between two operators, so
that the other tasks of an [embassy](https://embassy.dev), RTIC or hand-written executor run in
between:

```rust,ignore
#[embassy_executor::task]
async fn inference(mut ctx: ExecutionContext<'static, 'static, &'static Arena<'static>, &'static Arena<'static>>) {
    loop {
        let features = FEATURES.receive().await;
        ctx.input_mut(0).unwrap().as_mut_slice::<i8>().unwrap().copy_from_slice(&features);
        ctx.compute_async().await.unwrap();
        RESULTS.send(argmax(ctx.output(0).unwrap().as_slice::<i8>().unwrap())).await;
    }
}
```

`compute_async` depends on no executor: it only wakes itself before returning `Pending`.
Dropping the future abandons the inference.

## Stop at a deadline

An [`Observer`](write-a-backend.md) can return `ControlFlow::Break` from `before` to stop the
computation before the next operator. `compute_with`, `step_with` and `compute_async_with` then
return an error with code `ErrorCode::Timeout` and detail `Detail::Interrupted { op_index }`:

```rust,ignore
use core::ops::ControlFlow;
use nanoglia::{Observer, OpInfo};

/// Stops the inference once `deadline` (in cycles) has passed.
struct Deadline<F: FnMut() -> u32> {
    clock: F,
    deadline: u32,
}

impl<F: FnMut() -> u32> Observer for Deadline<F> {
    fn before(&mut self, _op: &OpInfo<'_>) -> ControlFlow<()> {
        if (self.clock)().wrapping_sub(self.deadline) as i32 >= 0 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}
```

The context stays usable after an interruption: write the inputs again and compute.

## See also

- [Write a backend](write-a-backend.md): observers that time each operator.
- [Errors](../reference/errors.md): `Detail::Interrupted`.
