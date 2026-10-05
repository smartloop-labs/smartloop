//! Port of `smartloop/model_host/__init__.py`.

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
