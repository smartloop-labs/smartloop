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
from smartloop.mcp.definitions import MCPDefinition, definition_for

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
