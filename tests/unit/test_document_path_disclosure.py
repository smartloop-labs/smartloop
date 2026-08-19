"""Document paths leave the API relative, never absolute.

An absolute path pins a client to one machine's layout: it breaks the moment
``<home_dir>`` is copied elsewhere or relocated into the macOS App Sandbox
container, and it discloses the operator's real home directory to every API
consumer.
"""

import os

import pytest

from smartloop.workspace.types import Document, relative_document_path


@pytest.fixture
def project_dir(tmp_path):
    d = tmp_path / "home" / "ws-1" / "proj-1"
    (d / "documents").mkdir(parents=True)
    (d / "documents" / "Invoice.pdf").write_bytes(b"%PDF")
    return d


def _doc(project_dir, name="Invoice.pdf"):
    return Document(id="d1", path=str(project_dir / "documents" / name),
                    name=name, status="indexed")


def test_path_is_relative_to_the_project(project_dir):
    assert _doc(project_dir).to_info(str(project_dir)).path == "documents/Invoice.pdf"


def test_absolute_path_never_escapes(project_dir):
    info = _doc(project_dir).to_info(str(project_dir))
    assert not os.path.isabs(info.path)
    assert str(project_dir) not in info.path


def test_home_directory_is_not_disclosed(project_dir, monkeypatch):
    """Even without a project_dir, the home prefix is stripped."""
    monkeypatch.setenv("SLP_HOME", str(project_dir.parent.parent))
    info = _doc(project_dir).to_info()
    assert info.path == "ws-1/proj-1/documents/Invoice.pdf"
    assert "home" not in info.path.split("/")[0]


def test_filename_is_still_reported(project_dir):
    assert _doc(project_dir).to_info(str(project_dir)).filename == "Invoice.pdf"


def test_path_outside_the_framework_degrades_to_a_bare_name(project_dir, tmp_path):
    """A path from somewhere else must not disclose its location."""
    outside = tmp_path / "elsewhere" / "secret-folder" / "leak.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x")
    rendered = relative_document_path(str(outside), str(project_dir))
    assert rendered == "leak.pdf"
    assert "secret-folder" not in rendered


def test_none_and_empty_pass_through():
    assert relative_document_path(None, "/x") is None
    assert relative_document_path("", "/x") == ""


def test_posix_separators_always(project_dir):
    """A path rendered on Windows must compare equal to one from macOS."""
    assert "\\" not in _doc(project_dir).to_info(str(project_dir)).path


def test_relative_form_survives_relocation(tmp_path):
    """The same document under a different home renders identically."""
    rendered = []
    for home in ("home_a", "home_b"):
        d = tmp_path / home / "ws-1" / "proj-1"
        (d / "documents").mkdir(parents=True)
        (d / "documents" / "Invoice.pdf").write_bytes(b"%PDF")
        rendered.append(_doc(d).to_info(str(d)).path)
    assert rendered[0] == rendered[1] == "documents/Invoice.pdf"


def test_missing_file_still_renders_relative(project_dir):
    """Rendering must not depend on the file existing on disk."""
    doc = Document(id="d2", path=str(project_dir / "documents" / "gone.pdf"),
                   name="gone.pdf", status="indexed")
    assert doc.to_info(str(project_dir)).path == "documents/gone.pdf"
