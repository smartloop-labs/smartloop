"""The generative model, factored out of the per-project process.

See :mod:`smartloop.model_host.client` for the interface a project agent
reaches the model through, and :mod:`smartloop.model_host.host` for the
process that owns it.
"""

from smartloop.model_host.client import (
    InProcessModelClient,
    ModelClient,
    ModelHostError,
    RemoteModelClient,
    default_socket_path,
    get_model_client,
    parse_address,
)
from smartloop.model_host.host import ModelHost, serve

__all__ = [
    "InProcessModelClient",
    "ModelClient",
    "ModelHost",
    "ModelHostError",
    "RemoteModelClient",
    "default_socket_path",
    "get_model_client",
    "parse_address",
    "serve",
]
