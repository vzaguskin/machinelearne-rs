//! Build script for machinelearne-rs library.
//!
//! Configures build-time settings for optional features.

fn main() {
    // Configure ort-sys to use system ONNX Runtime from /usr/local.
    // This only has an effect when the onnx-inference feature is enabled.
    println!("cargo:rustc-env=ORT_LIB_LOCATION=/usr/local");
    // Add library search path for linking
    println!("cargo:rustc-link-search=native=/usr/local/lib");

    // Re-run if features change
    println!("cargo:rerun-if-changed=Cargo.toml");
}
