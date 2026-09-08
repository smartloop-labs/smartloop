"""One supervised process per project, behind the port clients already use.

See :mod:`smartloop.supervisor.service` for the switch that turns this on,
:mod:`smartloop.supervisor.pool` for the child lifecycle, and
:mod:`smartloop.supervisor.routing` for how a request reaches the right one,
and :mod:`smartloop.supervisor.parent_watch` for what stops a child from
outliving the supervisor that spawned it.
"""

from smartloop.supervisor.parent_watch import (
    ParentWatch, parent_environment, start_parent_watch,
)
from smartloop.supervisor.pool import ChildStartError, ProjectAgentPool
from smartloop.supervisor.registry import (
    AgentRecord, AgentRegistry, process_alive, read_project_agent,
)
from smartloop.supervisor.routing import ProjectRoutingMiddleware
from smartloop.supervisor.service import (
    Supervisor, get_pool, get_supervisor, model_host_enabled, start_supervisor,
    stop_supervisor, supervision_enabled,
)

__all__ = [
    "AgentRecord",
    "AgentRegistry",
    "ChildStartError",
    "ParentWatch",
    "ProjectAgentPool",
    "ProjectRoutingMiddleware",
    "Supervisor",
    "get_pool",
    "get_supervisor",
    "model_host_enabled",
    "parent_environment",
    "process_alive",
    "read_project_agent",
    "start_parent_watch",
    "start_supervisor",
    "stop_supervisor",
    "supervision_enabled",
]
