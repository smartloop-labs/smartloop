import os
import json
import uuid
import logging
from pathlib import Path

from typing import Annotated, Optional, TypeVar, Type, Union
from pydantic import BaseModel, Field, ConfigDict
from rich.console import Console

from smartloop.auth import credential_store
from smartloop.model.downloader import SLPArtifactDownloader, HFDownloader
from smartloop.utils.log_utils import ConsoleLogger
from smartloop.config.app_settings import AppSettings
from smartloop.config.model_config import ModelConfig
from smartloop.model.slp_model import SLPModel
from smartloop.llms.base_llm import BaseLLM
from smartloop.workspace import Workspace, Project

_settings = AppSettings()

T = TypeVar('T', bound=SLPModel)

class Model(BaseModel):
    id: str
    workspace: Workspace
    slp_model: Annotated[Optional[SLPModel], Field(default=None, exclude=True)]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def save(self):
        """Persist workspace state (delegates to Workspace.save)."""
        self.workspace.save()

    def init(self):
        self.slp_model.init()

    def add_source(self, source: Union[str, Path], progress_callback=None):
        """Add a source file or URL to the project"""
        new_docs = self.slp_model.add_document(source, progress_callback=progress_callback)
        self.save()
        return new_docs

    def add_skill(self, name: str, content: str):
        """Add a skill to the current project"""
        self.slp_model.add_skill(name, content)
        self.save()

    def status(self):
        self.slp_model.status()

    def train(self, progress_callback=None):
        self.slp_model.train(progress_callback=progress_callback)
        self.save()

    def build(self, progress_callback=None):
        self.slp_model.build(progress_callback=progress_callback)

    def get_instance(self, debug: bool = False) -> BaseLLM:
        return self.slp_model.get_instance(debug=debug)


class ModelManager:

    @staticmethod
    def _load_model(t: Type[T], config: ModelConfig) -> Model:
        if not os.path.exists(Workspace.config_file):
            return ModelManager._create_new_model(t, config)

        try:
            with open(Workspace.config_file, "r") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Config file is empty")
                data = json.loads(content)
                # Remove computed fields that shouldn't be in __init__
                if 'workspace' in data:
                    data['workspace'].pop('model_dir', None)
                    data['workspace'].pop('model_path', None)
                    data['workspace'].pop('current_project', None)
                    data['workspace'].pop('model_id', None)
                model = Model.model_validate(data)

                # Re-resolve workspace paths against the current base_dir
                workspace_dir = os.path.join(_settings.home_dir, model.workspace.id)
                model.workspace.workspace_dir = workspace_dir
                model.workspace.models_dir = os.path.join(workspace_dir, "models")

                # Restore runtime state from the active WorkspaceModel
                active_ws_model = model.workspace.get_model(config.model_name)
                if active_ws_model:
                    if active_ws_model.config:
                        merged = active_ws_model.config.model_dump()
                        for key, value in config.model_dump().items():
                            if value is not None:
                                merged[key] = value
                        active_ws_model.config = ModelConfig.model_validate(merged)
                    else:
                        active_ws_model.config = config
                # Merge config only — don't change workspace active model

                return model
        except (json.JSONDecodeError, ValueError) as e:
            # If config file is corrupted or empty, create a new model
            return ModelManager._create_new_model(t, config)

    @staticmethod
    def get_model(t: Type[T], workspace: Workspace, logger: logging.Logger = ConsoleLogger(Console()), config: Optional[ModelConfig] = None) -> Model:
        """Return a Model backed by the given workspace.

        The workspace must already be initialised (via ``Workspace.init``).
        This method registers the model and attaches a downloader and SLPModel
        instance.  Callers are responsible for calling ``model.save()`` after
        their operation succeeds.
        """
        try:
            # Use the passed config; fall back to the active workspace model's config
            if config is None:
                ws_model = workspace.get_model()
                config = ws_model.config if ws_model else None
            if config is None:
                raise ValueError("No model config available — pass config explicitly or ensure workspace has an active model.")

            # Ensure the model is registered (without changing workspace active model)
            ws_model = workspace.get_model(config.model_name)
            if ws_model is None:
                ws_model = workspace.add_model(config.model_name, t.model_id(), config=config)

            # Switch to requested project if specified in config
            if config.project_id and workspace.projects:
                target = next(
                    (p for p in workspace.projects if p.id == config.project_id),
                    None,
                )
                if target is not None:
                    for p in workspace.projects:
                        p.current = (p.id == target.id)

            # Load model id from persisted config, or generate a new one
            if os.path.exists(Workspace.config_file):
                with open(Workspace.config_file, "r") as f:
                    raw = f.read().strip()
                if raw:
                    try:
                        data = json.loads(raw)
                        model_id = data.get("id", str(uuid.uuid4()))
                    except json.JSONDecodeError:
                        logger.warning("config.json contains invalid JSON; generating new model id")
                        model_id = str(uuid.uuid4())
                else:
                    model_id = str(uuid.uuid4())
            else:
                model_id = str(uuid.uuid4())

            # Attach the downloader + SLPModel instance
            downloader_cls = SLPArtifactDownloader if credential_store.has_credentials() else (HFDownloader if config.hf_token else SLPArtifactDownloader)
            downloader = downloader_cls(workspace, logger)
            slp_model = t(workspace, downloader, logger)

            model = Model(id=model_id, workspace=workspace, slp_model=slp_model)
        except Exception as e:
            logger.error(f"Failed to initialise model: {e}")
            raise

        return model



