"""A connection described by a file: what it says, and how it is launched.

Dropping a ``.toml`` into ``~/.smartloop/mcp.d/`` is how a server is added
without the UI, so what the file can express is the whole feature.  Two things
matter here: the launcher belongs to the server (``uvx`` is the default for a
bare package name, not an assumption about every server), and one malformed
drop-in must not take the working connections down with it.

These files used to also carry crawl plans — declarative walks that vectorized
a server's pages ahead of time.  Crawling is gone; a definition is a connection
now, and nothing here indexes anything.
"""

import os

import pytest

from smartloop.mcp.definition_loader import (
    definitions,
    load_definitions,
    parse_definitions,
    reload_definitions,
    user_definition_paths,
)
from smartloop.mcp import definitions as definitions_module
from smartloop import paths as paths_module
from smartloop.mcp.definitions import MCPDefinition, definition_for, retired_fields

_MINIMAL = '''
[[mcp]]
name = "acme"
package = "acme-mcp"
'''


class TestConnections:
    def test_a_local_server_carries_its_command(self):
        text = """
[[mcp]]
name = "yahoo-finance"
package = "yahoo-finance-mcp"
"""
        d = parse_definitions(text, "test")["yahoo-finance"]
        assert d.launch == ("uvx", ["yahoo-finance-mcp"])
        assert d.package == "yahoo-finance-mcp"
        assert d.kind == "local"

    def test_a_remote_server_carries_its_url(self):
        text = """
[[mcp]]
name = "my-wiki"
url = "https://mcp.example.com/mcp"
"""
        d = parse_definitions(text, "test")["my-wiki"]
        assert d.url == "https://mcp.example.com/mcp"
        assert d.kind == "remote"

    def test_a_file_may_only_name_a_server(self):
        """No connection at all still matches an installed one, which is how a
        file dropped in later attaches to a server added through the UI."""
        d = parse_definitions('[[mcp]]\nname = "notion"\n', "test")["notion"]
        assert d.kind is None

    def test_the_name_becomes_the_key_and_a_match_term(self):
        text = """
[[mcp]]
name = "Yahoo-Finance"
match = ["yfinance"]
package = "yahoo-finance-mcp"
"""
        d = parse_definitions(text, "test")["yahoo-finance"]
        assert d.key == "yahoo-finance"
        assert "yfinance" in d.match_terms

    def test_a_misspelled_field_is_rejected(self):
        # A typo must name the field rather than being silently ignored.
        with pytest.raises(Exception, match="packag"):
            parse_definitions('[[mcp]]\nname = "x"\npackag = "y"\n', "test")


class TestFilesWrittenForTheOldIndexingFormat:
    """A file that configured crawling still names a working connection.

    Crawling is gone, so those fields describe nothing — but they sit beside a
    url or a package name that is unaffected, and someone's existing file has
    to keep installing rather than being rejected for saying something that is
    now a no-op.
    """

    def test_the_retired_indexing_fields_are_ignored(self):
        text = """
[[mcp]]
name = "stocks"
package = "yahoo-finance-mcp"
crawl = true
index_every = "15m"

[[mcp.index]]
label = "stock-info"
tools = ["get_stock_info"]
max_pages = 20
max_parents = 25
"""
        d = parse_definitions(text, "test")["stocks"]
        assert d.launch == ("uvx", ["yahoo-finance-mcp"])
        assert d.kind == "local"
        assert not hasattr(d, "index")

    def test_the_ignored_fields_are_named_in_the_log(self, caplog):
        text = '[[mcp]]\nname = "x"\nurl = "https://e.example/mcp"\nindex_every = "6h"\n'
        with caplog.at_level("WARNING"):
            parse_definitions(text, "some-file.toml")
        assert "index_every" in caplog.text
        assert "some-file.toml" in caplog.text

    def test_a_typo_is_still_rejected_beside_them(self):
        # Ignoring the retired keys must not turn into ignoring every unknown
        # key: `packag` still has to be reported.
        text = '[[mcp]]\nname = "x"\nindex_every = "6h"\npackag = "y"\n'
        with pytest.raises(Exception, match="packag"):
            parse_definitions(text, "test")

    def test_retired_fields_reports_what_an_entry_carries(self):
        assert retired_fields({"name": "x", "index_every": "1h", "index": []}) == (
            "index_every", "index",
        )
        assert retired_fields({"name": "x", "package": "y"}) == ()


class TestLaunchers:
    """`uvx` is the default for a bare name, not an assumption about all of
    them: most MCP servers ship on npm, and plenty are a module or a path."""

    def test_a_package_can_name_its_own_launcher(self):
        text = """
[[mcp]]
name = "node-server"
package = "npx -y @example/some-mcp"
"""
        d = parse_definitions(text, "test")["node-server"]
        assert d.kind == "local"
        assert d.launch == ("npx", ["-y", "@example/some-mcp"])

    def test_a_scoped_npm_name_is_run_with_npx(self):
        # `@scope/pkg` cannot be a PyPI package, so uvx would only ever fail.
        text = """
[[mcp]]
name = "filesystem"
package = "@modelcontextprotocol/server-filesystem"
"""
        d = parse_definitions(text, "test")["filesystem"]
        assert d.launch == ("npx", ["-y", "@modelcontextprotocol/server-filesystem"])

    def test_a_module_or_a_path_is_a_command_line_too(self):
        text = """
[[mcp]]
name = "local-module"
package = "python -m my.server --port 0"
"""
        d = parse_definitions(text, "test")["local-module"]
        assert d.launch == ("python", ["-m", "my.server", "--port", "0"])

    def test_a_command_is_there_for_what_a_package_name_cannot_say(self):
        text = """
[[mcp]]
name = "custom"
command = "npx"
args = ["-y", "some-mcp"]
"""
        d = parse_definitions(text, "test")["custom"]
        assert d.launch == ("npx", ["-y", "some-mcp"])

    def test_a_command_may_carry_its_own_arguments(self):
        text = """
[[mcp]]
name = "inline"
command = "uv tool run --from pkg pkg"
"""
        d = parse_definitions(text, "test")["inline"]
        assert d.launch == ("uv", ["tool", "run", "--from", "pkg", "pkg"])

    def test_a_command_expands_home_directory(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        d = MCPDefinition(name="local", command="~/.local/craft mcp")
        assert d.launch == ("/home/tester/.local/craft", ["mcp"])

    def test_a_tilde_reads_past_an_app_sandbox_container(self, monkeypatch):
        # Inside the macOS App Sandbox $HOME is the container, not the user's
        # home, so `~/.local/bin/craft` expanded to a path under
        # .../Containers/<id>/Data that nothing had ever installed into — the
        # ENOENT every local server added to the sandboxed build failed with.
        monkeypatch.setenv(
            "HOME", "/Users/tester/Library/Containers/ai.smartloop.studio/Data",
        )
        monkeypatch.setattr(
            paths_module, "real_home", lambda: "/Users/tester",
        )
        d = MCPDefinition(name="local", command="~/.local/bin/craft mcp")
        assert d.launch == ("/Users/tester/.local/bin/craft", ["mcp"])

    def test_args_may_be_written_as_one_string(self):
        # How everyone writes a command down, and how every other MCP client
        # displays one; iterating it per character is the alternative.
        text = """
[[mcp]]
name = "stringy"
command = "npx"
args = "-y some-mcp"
"""
        d = parse_definitions(text, "test")["stringy"]
        assert d.launch == ("npx", ["-y", "some-mcp"])

    def test_a_quoted_argument_survives_as_one_argument(self):
        d = MCPDefinition(name="q", command='python -c "import x; x.main()"')
        assert d.launch == ("python", ["-c", "import x; x.main()"])

    def test_args_without_a_command_is_an_error_naming_the_field(self):
        text = """
[[mcp]]
name = "orphan"
args = ["-y", "some-mcp"]
"""
        with pytest.raises(Exception, match="args without a command"):
            parse_definitions(text, "test")

    def test_a_package_and_a_command_say_the_same_thing_twice(self):
        text = """
[[mcp]]
name = "confused"
package = "x"
command = "npx"
"""
        with pytest.raises(Exception, match="twice"):
            parse_definitions(text, "test")

    def test_a_server_is_local_or_remote_not_both(self):
        text = """
[[mcp]]
name = "confused"
package = "x"
url = "https://x"
"""
        with pytest.raises(Exception, match="alternatives"):
            parse_definitions(text, "test")


class TestUserFiles:
    def _write(self, home, name, text, in_dropin_dir=True):
        directory = os.path.join(home, "mcp.d") if in_dropin_dir else home
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def test_a_dropped_in_file_is_loaded(self, tmp_path):
        self._write(str(tmp_path), "acme.toml", _MINIMAL)
        assert "acme" in load_definitions(home_dir=str(tmp_path))

    def test_a_single_user_file_also_works(self, tmp_path):
        self._write(str(tmp_path), "mcp.toml", _MINIMAL, in_dropin_dir=False)
        assert "acme" in load_definitions(home_dir=str(tmp_path))

    def test_a_malformed_drop_in_does_not_take_the_others_down(self, tmp_path):
        self._write(str(tmp_path), "aaa-broken.toml", "this is not [valid toml")
        self._write(str(tmp_path), "bbb-good.toml", _MINIMAL)
        assert "acme" in load_definitions(home_dir=str(tmp_path))

    def test_drop_ins_are_read_in_name_order(self, tmp_path):
        self._write(str(tmp_path), "10-first.toml", _MINIMAL)
        self._write(str(tmp_path), "20-second.toml", _MINIMAL.replace(
            'package = "acme-mcp"', 'package = "npx -y acme-mcp"',
        ))
        # Later file wins for the same key.
        loaded = load_definitions(home_dir=str(tmp_path))["acme"]
        assert loaded.launch == ("npx", ["-y", "acme-mcp"])

    def test_user_paths_list_the_single_file_before_the_directory(self, tmp_path):
        self._write(str(tmp_path), "mcp.toml", _MINIMAL, in_dropin_dir=False)
        self._write(str(tmp_path), "z.toml", _MINIMAL)
        paths = user_definition_paths(str(tmp_path))
        assert paths[0].endswith("mcp.toml")
        assert paths[1].endswith(os.path.join("mcp.d", "z.toml"))

    def test_no_user_files_is_not_an_error(self, tmp_path):
        assert user_definition_paths(str(tmp_path)) == []
        assert load_definitions(home_dir=str(tmp_path)) == {}


class TestCache:
    def test_the_registry_is_cached_and_reloadable(self):
        first = definitions()
        assert definitions() is first
        assert reload_definitions() is not first


class TestFindingTheFileForAServer:
    """Deleting a connection has to delete the file that defined it, or
    re-adding a server by the same name silently inherits the old command."""

    class _Server:
        def __init__(self, name="acme", catalog_slug=None, pkg=None):
            self.name = name
            self.catalog_slug = catalog_slug
            self.pkg = pkg
            self.remote_config = None
            self.config = None

    def test_the_catalog_slug_matches_first(self, monkeypatch):
        found = parse_definitions(_MINIMAL, "test")
        monkeypatch.setattr(
            "smartloop.mcp.definition_loader.definitions", lambda: found,
        )
        server = self._Server(name="Something Else", catalog_slug="acme")
        assert definition_for(server) is found["acme"]

    def test_a_declared_match_term_finds_it(self, monkeypatch):
        found = parse_definitions(
            '[[mcp]]\nname = "acme"\nmatch = ["yfinance"]\npackage = "x"\n', "test",
        )
        monkeypatch.setattr(
            "smartloop.mcp.definition_loader.definitions", lambda: found,
        )
        assert definition_for(self._Server(name="YFinance tools")) is found["acme"]

    def test_an_unrelated_server_matches_nothing(self, monkeypatch):
        found = parse_definitions(_MINIMAL, "test")
        monkeypatch.setattr(
            "smartloop.mcp.definition_loader.definitions", lambda: found,
        )
        assert definition_for(self._Server(name="notion")) is None


class TestNothingRewritesTheCommand:
    """What a connection configures is what runs.

    Registration used to fold `uvx <pkg>` into `uvx --from <pkg> <pkg>` and
    swap `uvx` for `uv tool run` — so the command in the config was not the
    command that ran, invisibly, and only ever correctly for uv.
    """

    def _register(self, command, args):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from smartloop.api.services.model_service import ModelService

        project = SimpleNamespace(
            id="p1", name="proj", project_dir="/tmp/p1", mcp_servers=[],
        )
        svc = MagicMock()
        svc._get_project = MagicMock(return_value=project)
        svc._model = MagicMock()
        svc._mcp_server_to_dict = ModelService._mcp_server_to_dict
        with patch("smartloop.api.services.model_service.LocalMCPClient"):
            ModelService.add_local_mcp_server(
                svc, command=command, args=args, project_id="p1",
            )
        return project.mcp_servers[0].config

    def test_a_bare_uvx_package_is_stored_as_written(self):
        config = self._register("uvx", ["yahoo-finance-mcp"])
        assert (config.command, config.args) == ("uvx", ["yahoo-finance-mcp"])

    def test_an_npx_invocation_is_stored_as_written(self):
        config = self._register("npx", ["-y", "@example/some-mcp"])
        assert (config.command, config.args) == ("npx", ["-y", "@example/some-mcp"])

    def test_a_command_line_in_one_string_is_only_split(self):
        config = self._register("uv tool run --from pkg pkg", [])
        assert (config.command, config.args) == (
            "uv", ["tool", "run", "--from", "pkg", "pkg"],
        )

    def test_a_program_path_needs_no_arguments(self):
        config = self._register("/opt/bin/my-mcp", [])
        assert (config.command, config.args) == ("/opt/bin/my-mcp", [])


class TestTheCommandIsResolvedToItsPath:
    """Which `uvx` — the app's PATH is not the shell's, so a command a user
    verified in a terminal can be missing from the process that launches it."""

    def _client(self, command, args=("--from", "pkg", "pkg")):
        from smartloop.mcp.local_client import LocalMCPClient

        return LocalMCPClient(command=command, args=list(args))

    def test_a_command_on_the_path_resolves_to_its_full_path(self, tmp_path):
        program = tmp_path / "uvx"
        program.write_text("#!/bin/sh\n")
        program.chmod(0o755)

        resolved = self._client("uvx")._resolve_command({"PATH": str(tmp_path)})

        assert resolved == str(program)

    def test_an_absolute_path_is_used_as_given(self):
        assert self._client("/opt/bin/my-mcp", ())._resolve_command({}) == "/opt/bin/my-mcp"

    def test_a_command_nowhere_on_the_path_is_left_to_the_os(self):
        # The OS produces the error rather than this inventing one.
        assert self._client("definitely-not-installed")._resolve_command(
            {"PATH": "/nonexistent"}
        ) == "definitely-not-installed"

    def test_a_built_in_server_runs_under_the_current_runtime(self):
        import sys

        client = self._client("python", ("-m", "smartloop.api.mcp.system_server"))
        assert client._resolve_command({}) == sys.executable
