//! Detects whether the target has the Arm `VFMA.F32` instruction, used for the fused
//! multiply-adds of the float convolutions (`cfg(nanoglia_arm_vfma)`).

fn main() {
    println!("cargo::rustc-check-cfg=cfg(nanoglia_arm_vfma)");
    println!("cargo::rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    // The hard-float Arm M-profile targets have an FPv4 FPU or later, all with VFMA.
    let m_profile = ["thumbv7em-", "thumbv8m.main-", "thumbv8.1m.main-"]
        .iter()
        .any(|prefix| target.starts_with(prefix));
    if m_profile && target.ends_with("eabihf") {
        println!("cargo::rustc-cfg=nanoglia_arm_vfma");
    }
}
