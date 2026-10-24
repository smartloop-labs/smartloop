//! Port of `smartloop/model/__init__.py`.

pub mod downloader;
pub mod generic;
pub mod model_manager;
pub mod slp_model;
pub mod spec;
pub mod workers;

pub use slp_model::SLPModel;
