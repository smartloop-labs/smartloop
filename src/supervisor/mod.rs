//! Port of `smartloop/supervisor/__init__.py`.
//!
//! One supervised process per project, behind the port clients already use.
//! See [`service`] for the switch that turns this on, [`pool`] for the child
//! lifecycle, [`routing`] for how a request reaches the right one, and
//! [`parent_watch`] for what stops a child from outliving its supervisor.

pub mod parent_watch;
pub mod pool;
pub mod registry;
pub mod routing;
pub mod service;
pub mod spawn;

pub use parent_watch::{parent_environment, start_parent_watch, ParentWatch};
pub use pool::{ChildStartError, ProjectAgentPool};
pub use registry::{process_alive, read_project_agent, AgentRecord, AgentRegistry};
pub use routing::ProjectRoutingMiddleware;
pub use service::{
    get_pool, get_supervisor, model_host_enabled, start_supervisor, stop_supervisor,
    supervision_enabled, Supervisor,
};
