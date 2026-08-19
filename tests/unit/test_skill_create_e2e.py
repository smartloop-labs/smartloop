"""E2E tests: skill creation writes correct front matter and directory structure."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.workspace.workspace import Project, Skill


@pytest.fixture()
def skill_workspace(tmp_path):
    """The project's state dir — where skills are written.

    Skills hang off ``project_dir``, never a client-supplied path, so this
    fixture only has to hand out a temp state dir.  A ``cwd`` pointing
    somewhere else entirely is set on the projects below to prove it is
    ignored.
    """
    state_dir = tmp_path / "state" / "proj1"
    state_dir.mkdir(parents=True)
    return state_dir


class TestWriteSkillMd:
    @staticmethod
    def _project(state_dir, name="my-project"):
        return Project(
            id="proj1", project_dir=str(state_dir), current=True, name=name,
            cwd="/somewhere/else/entirely",
        )

    def test_creates_skills_dir(self, skill_workspace):
        skill = Skill(id="abc123", name="code-review", content="Review the code carefully.")
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        skills_dir = skill_workspace / "skills"
        assert skills_dir.is_dir()

    def test_creates_skill_subdir(self, skill_workspace):
        skill = Skill(id="abc123", name="code-review", content="Review the code carefully.")
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        skill_dir = skill_workspace / "skills" / "code-review"
        assert skill_dir.is_dir()

    def test_creates_skill_md_file(self, skill_workspace):
        skill = Skill(id="abc123", name="code-review", content="Review the code carefully.")
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        skill_path = skill_workspace / "skills" / "code-review" / "SKILL.md"
        assert skill_path.is_file()

    def test_front_matter_has_name(self, skill_workspace):
        skill = Skill(id="abc123", name="code-review", content="Review the code carefully.")
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        content = (skill_workspace / "skills" / "code-review" / "SKILL.md").read_text()
        assert "---" in content
        assert "name: code-review" in content

    def test_front_matter_has_description(self, skill_workspace):
        skill = Skill(id="abc123", name="code-review", description="Does a thorough code review.", content="Review the code carefully.")
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        content = (skill_workspace / "skills" / "code-review" / "SKILL.md").read_text()
        assert "description: Does a thorough code review." in content

    def test_front_matter_omits_description_when_none(self, skill_workspace):
        skill = Skill(id="abc123", name="code-review", content="Review the code carefully.")
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        content = (skill_workspace / "skills" / "code-review" / "SKILL.md").read_text()
        assert "description:" not in content

    def test_content_written_after_front_matter(self, skill_workspace):
        body = "Review the code carefully."
        skill = Skill(id="abc123", name="code-review", content=body)
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        content = (skill_workspace / "skills" / "code-review" / "SKILL.md").read_text()
        # front matter ends with --- then a blank line before body
        assert content.endswith(f"---\n\n{body}")

    def test_front_matter_delimiters_present(self, skill_workspace):
        skill = Skill(id="abc123", name="greet", content="Say hello.")
        ModelService._write_skill_md(self._project(skill_workspace), skill)

        content = (skill_workspace / "skills" / "greet" / "SKILL.md").read_text()
        lines = content.splitlines()
        assert lines[0] == "---"
        closing = lines.index("---", 1)
        assert closing > 0


class TestAddSkillCreatesFile:
    """Integration-level: add_skill on ModelService writes SKILL.md with front matter."""

    def _make_service(self, state_dir):
        """Return a ModelService with its internal state mocked just enough."""
        svc = object.__new__(ModelService)
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value='{"name": "greet", "body": "Greet the user politely."}')
        mock_llm.get_max_tokens = MagicMock(return_value=4096)
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        svc._llm = mock_llm
        svc._large_llm = None
        svc._large_model_attempted = True
        svc._model = MagicMock()
        svc._inference_lock = __import__("threading").Lock()
        project = Project(
            id="proj1",
            project_dir=str(state_dir),
            current=True,
            name="my-project",
            cwd="/somewhere/else/entirely",
        )
        svc._model.workspace.projects = [project]
        svc._model.save = MagicMock()
        return svc, project

    def test_add_skill_creates_skills_dir(self, skill_workspace):
        svc, project = self._make_service(skill_workspace)

        with patch.object(ModelService, "_get_project", return_value=project):
            svc.add_skill("proj1", content="Greet the user politely.", name="greet", description="Greeter")

        skills_dir = skill_workspace / "skills"
        assert skills_dir.is_dir()

    def test_add_skill_creates_skill_md(self, skill_workspace):
        svc, project = self._make_service(skill_workspace)

        with patch.object(ModelService, "_get_project", return_value=project):
            result = svc.add_skill("proj1", content="Greet the user politely.", name="greet", description="Greeter")

        skill_path = skill_workspace / "skills" / "greet" / "SKILL.md"
        assert skill_path.is_file()
        assert result["name"] == "greet"

    def test_add_skill_front_matter_name(self, skill_workspace):
        svc, project = self._make_service(skill_workspace)

        with patch.object(ModelService, "_get_project", return_value=project):
            svc.add_skill("proj1", content="Greet the user.", name="greet", description="Friendly greeter")

        content = (skill_workspace / "skills" / "greet" / "SKILL.md").read_text()
        assert "name: greet" in content
        assert "description: Friendly greeter" in content

    def test_add_skill_sanitizes_name(self, skill_workspace):
        svc, project = self._make_service(skill_workspace)

        with patch.object(ModelService, "_get_project", return_value=project):
            result = svc.add_skill("proj1", content="Do stuff.", name="My Cool Skill!", description="Does stuff")

        assert result["name"] == "my-cool-skill"
        skill_path = skill_workspace / "skills" / "my-cool-skill" / "SKILL.md"
        assert skill_path.is_file()
