//! Generates the model-host gRPC stubs (replaces the checked-in
//! `model_host_pb2*.py`) from `proto/model_host.proto`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("PROTOC").is_none() {
        // SAFETY: build scripts are single-threaded.
        unsafe { std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?) };
    }
    println!("cargo:rerun-if-changed=proto/model_host.proto");
    tonic_prost_build::configure().compile_protos(&["proto/model_host.proto"], &["proto"])?;
    Ok(())
}
