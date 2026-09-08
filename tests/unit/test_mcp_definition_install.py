"""Installing a connection from a dropped-in TOML file.

The file is the whole definition — what the server is called and how to reach
it. Two things have to land: the file on disk under ``mcp.d`` so the connection
can be edited or removed as a file, and the connection itself through the
ordinary registration path so an OAuth server behaves exactly as a typed-in URL
does.

Most of what's pinned here is the file being rejected, because a dropped file
is the one input nobody proofreads.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from smartloop.api.model import MCPDefinitionRequest
from smartloop.api.routers.mcp_servers import register_mcp_from_definition

_LOCAL = '''
[[mcp]]
name = "yahoo-finance"
package = "yahoo-finance-mcp"
env = { TZ = "UTC" }
'''

_REMOTE = '''
[[mcp]]
name = "my-wiki"
url = "https://mcp.example.com/mcp"
'''


async def _install(toml, home, filename="dropped.toml"):
    """Run the endpoint with registration stubbed, returning (result, request)."""
    registered = {}

    async def _fake_register(project_id, request, http_request, model_service):
        registered["request"] = request
        return {"auth_type": "none", "id": "srv1"}

    with patch("smartloop.config.app_settings.AppSettings") as settings, \
         patch("smartloop.api.routers.mcp_servers.register_mcp_server", _fake_register):
        settings.return_value.home_dir = home
        result = await register_mcp_from_definition(
            project_id="p1",
            request=MCPDefinitionRequest(toml=toml, filename=filename),
            http_request=None,
            model_service=MagicMock(),
        )
    return result, registered.get("request")


class TestInstall:
    @pytest.mark.asyncio
    async def test_a_local_definition_registers_its_command(self, tmp_path):
        _, request = await _install(_LOCAL, str(tmp_path))
        assert request.server_type == "local"
        assert request.command == "uvx"
        assert request.args == ["yahoo-finance-mcp"]
        assert request.name == "yahoo-finance"

    @pytest.mark.asyncio
    async def test_a_remote_definition_registers_its_url(self, tmp_path):
        _, request = await _install(_REMOTE, str(tmp_path))
        assert request.server_type == "remote"
        assert request.server_url == "https://mcp.example.com/mcp"
        # So an OAuth server can hand back to the app, same as a typed URL.
        assert request.return_url == "smartloop://mcp-authorization"

    @pytest.mark.asyncio
    async def test_the_file_is_on_disk_before_registration_returns(self, tmp_path):
        # Written before the registration it describes, so a failed
        # registration leaves something to retry from rather than nothing.
        await _install(_LOCAL, str(tmp_path))
        saved = tmp_path / "mcp.d" / "yahoo-finance.toml"
        assert saved.is_file()
        assert "yahoo-finance-mcp" in saved.read_text()

    @pytest.mark.asyncio
    async def test_the_saved_file_is_named_after_the_connection(self, tmp_path):
        # Not after whatever it was called on the way in — re-dropping an
        # edited file has to replace it, not stack a second copy beside it.
        await _install(_LOCAL, str(tmp_path), filename="my-download (3).toml")
        assert os.listdir(tmp_path / "mcp.d") == ["yahoo-finance.toml"]

    @pytest.mark.asyncio
    async def test_re_dropping_an_edited_file_replaces_it(self, tmp_path):
        await _install(_LOCAL, str(tmp_path))
        edited = _LOCAL.replace('TZ = "UTC"', 'TZ = "Europe/Berlin"')
        await _install(edited, str(tmp_path))
        saved = (tmp_path / "mcp.d" / "yahoo-finance.toml").read_text()
        assert "Europe/Berlin" in saved and "UTC" not in saved
        assert os.listdir(tmp_path / "mcp.d") == ["yahoo-finance.toml"]


class TestRejections:
    @pytest.mark.asyncio
    async def test_a_file_that_is_not_toml_is_refused(self, tmp_path):
        with pytest.raises(HTTPException) as e:
            await _install("this is not [valid toml", str(tmp_path))
        assert e.value.status_code == 400

    @pytest.mark.asyncio
    async def test_a_file_with_no_mcp_entry_says_so(self, tmp_path):
        with pytest.raises(HTTPException) as e:
            await _install('[something_else]\nfoo = "bar"\n', str(tmp_path))
        assert "[[mcp]]" in e.value.detail

    @pytest.mark.asyncio
    async def test_a_definition_with_no_connection_is_refused(self, tmp_path):
        # A name alone can't be installed — there is nothing to install.
        toml = '''
[[mcp]]
name = "orphan"
'''
        with pytest.raises(HTTPException) as e:
            await _install(toml, str(tmp_path))
        assert "no connection" in e.value.detail

    @pytest.mark.asyncio
    async def test_a_file_describing_several_servers_is_refused(self, tmp_path):
        # One file per connection: the file is saved under the connection's
        # name, and two connections have no single name to save it under.
        with pytest.raises(HTTPException) as e:
            await _install(_LOCAL + _REMOTE, str(tmp_path))
        assert "one file per connection" in e.value.detail

    @pytest.mark.asyncio
    async def test_a_misspelled_field_fails_the_whole_file(self, tmp_path):
        # Silently ignoring it would install a connection missing the thing the
        # author wrote the file to set.
        toml = '''
[[mcp]]
name = "acme"
packag = "acme-mcp"
'''
        with pytest.raises(HTTPException) as e:
            await _install(toml, str(tmp_path))
        assert e.value.status_code == 400
        assert not (tmp_path / "mcp.d").exists()

    @pytest.mark.asyncio
    async def test_an_unwritable_drop_in_dir_still_installs_the_connection(self, tmp_path):
        # Keeping the file is a bonus; the connection is the point, and losing
        # the first must not cost the second.
        home = tmp_path / "home"
        home.mkdir()
        (home / "mcp.d").write_text("not a directory")
        _, request = await _install(_LOCAL, str(home))
        assert request is not None
        assert request.command == "uvx"
