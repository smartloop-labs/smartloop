"""The orchestration timeout must not free models still being decoded on.

When the 180s budget expires the router signals ``cancelled`` and stops
waiting, but it deliberately does not await the executor future — a stuck
native llama.cpp call would block forever.  So the orchestration thread is
still running, very likely inside that native call.  Closing its llama.cpp
context there is a use-after-free that segfaults the whole server, which is
what ``make run-debug`` hit right after "[mem] Evicting prior-step capability
model qwen3.5-2b".  The unload has to wait for the thread to unwind.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from smartloop.api.routers.chat import _unload_when_idle


@pytest.mark.asyncio
async def test_models_are_not_freed_while_the_thread_is_running():
    loop = asyncio.get_running_loop()
    service = MagicMock()
    still_working = asyncio.Event()

    future = loop.create_future()
    _unload_when_idle(future, service, loop)

    # The orchestration thread has not unwound yet.
    await asyncio.sleep(0)
    service.unload_auxiliary_models.assert_not_called()
    assert not still_working.is_set()


@pytest.mark.asyncio
async def test_models_are_freed_once_the_thread_unwinds():
    loop = asyncio.get_running_loop()
    service = MagicMock()

    future = loop.create_future()
    _unload_when_idle(future, service, loop)

    future.set_result(("context", None, None, [], None, False))
    # One hop for the done-callback, one for the executor it hands off to.
    for _ in range(10):
        await asyncio.sleep(0.01)
        if service.unload_auxiliary_models.called:
            break

    service.unload_auxiliary_models.assert_called_once()


@pytest.mark.asyncio
async def test_an_unload_failure_is_swallowed():
    # Cleanup runs after the client already got its timeout response; a failure
    # here must not surface anywhere.
    loop = asyncio.get_running_loop()
    service = MagicMock()
    service.unload_auxiliary_models.side_effect = RuntimeError("GPU busy")

    future = loop.create_future()
    _unload_when_idle(future, service, loop)
    future.set_result(None)

    for _ in range(10):
        await asyncio.sleep(0.01)
        if service.unload_auxiliary_models.called:
            break
    assert service.unload_auxiliary_models.called
