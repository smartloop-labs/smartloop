//! Generates the model-host gRPC stubs (replaces the checked-in
//! `model_host_pb2*.py`) from `proto/model_host.proto`, and sets the `slp`
//! binary's link options for the packaged layout.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("PROTOC").is_none() {
        // SAFETY: build scripts are single-threaded.
        unsafe { std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?) };
    }
    println!("cargo:rerun-if-changed=proto/model_host.proto");
    tonic_prost_build::configure().compile_protos(&["proto/model_host.proto"], &["proto"])?;

    // Packaged builds (`make build-binary`) ship the Vulkan loader in
    // `<install dir>/lib/` next to `slp`, so end users need no Vulkan runtime
    // installed — only their GPU driver's ICD.  Search there first; when the
    // directory is absent (cargo run / cargo install) the dynamic loader falls
    // back to the system library paths.
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/lib"),
        // Leave room in the Mach-O header so `make build-binary` can rewrite a
        // Homebrew dylib reference to `@executable_path/lib/...` with
        // install_name_tool when the Vulkan backend is enabled.
        Ok("macos") => println!("cargo:rustc-link-arg-bins=-Wl,-headerpad_max_install_names"),
        _ => {}
    }
    Ok(())
}
