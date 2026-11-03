use nanoglia_core::backend::Backend;

/// The ordered list of backends a graph picks its kernels from: for each operator, the first
/// backend that supports it wins.
#[derive(Clone, Copy)]
pub struct Registry<'r> {
    backends: &'r [&'r dyn Backend],
}

impl core::fmt::Debug for Registry<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list()
            .entries(self.backends.iter().map(|b| b.name()))
            .finish()
    }
}

static DEFAULT_BACKENDS: &[&dyn Backend] = &[
    #[cfg(feature = "cmsis-nn")]
    &nanoglia_cmsis_nn::BACKEND,
    #[cfg(feature = "kernels")]
    &nanoglia_kernels::BACKEND,
];

impl<'r> Registry<'r> {
    /// A registry trying `backends` in order.
    pub const fn new(backends: &'r [&'r dyn Backend]) -> Self {
        Self { backends }
    }

    /// The backends.
    pub fn backends(&self) -> &'r [&'r dyn Backend] {
        self.backends
    }
}

impl Default for Registry<'static> {
    /// The backends enabled by the crate features, fastest first: CMSIS-NN (feature
    /// `cmsis-nn`), then the Rust kernels (feature `kernels`).
    fn default() -> Self {
        Self::new(DEFAULT_BACKENDS)
    }
}
