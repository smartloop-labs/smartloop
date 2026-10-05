//! Port of `smartloop/model_host/__init__.py`.
//!
//! The generative model, factored out of the per-project process.
//!
//! See [`client`] for the interface a project agent reaches the model through,
//! and [`host`] for the process that owns it.

pub mod entry;
pub mod bench;
pub mod client;
pub mod host;
pub mod remote_llm;

/// Generated gRPC/protobuf types for `proto/model_host.proto`
/// (replaces `model_host_pb2.py` / `model_host_pb2_grpc.py`).
pub mod pb {
    tonic::include_proto!("smartloop.model_host.v1");
}

pub use client::{
    default_socket_path, get_model_client, parse_address, InProcessModelClient, ModelClient,
    ModelHostError, RemoteModelClient,
};
pub use host::{serve, ModelHost};
