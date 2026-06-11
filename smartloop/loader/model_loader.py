import os
import time
import logging
from typing import Iterator, Optional
import torch

from smartloop.workspace import Workspace, Project
from smartloop.llms import PreTrainedLLM
from smartloop.utils.device_utils import get_device_config

from prettytable import PrettyTable
from rich.console import Console
from langchain_core.language_models.llms import LLM
from smartloop.llms.llm_factory import LlmFactory

console= Console()

class ModelLoader:
    debug: bool = False
    is_trainable: bool = False

    def __init__(self, workspace: Workspace, logger: logging.Logger, debug: bool = False, is_trainable: bool = False, gguf_path: Optional[str] = None):
        self.workspace = workspace
        self.logger = logger
        self.debug = debug
        self.is_trainable = is_trainable
        self.gguf_path = gguf_path

    def _resolve_adapter_path(self) -> str | None:
        """Find the GGUF adapter file for the current project, if one exists."""
        project = self.workspace.current_project
        adapters = project.adapters if project else set()
        if not adapters:
            return None

        for adapter_name in adapters:
            adapter_gguf = os.path.join(project.project_dir, "model", f"{adapter_name}.gguf")
            if os.path.exists(adapter_gguf):
                self.logger.info(f"Found LoRA adapter GGUF: {adapter_gguf}")
                return adapter_gguf

        return None

    def load(self) -> LLM:
        try:
            self.logger.info(" Preparing and loading model...")
            
            if self.debug:
                from smartloop_trainer.causal_lm import SLPModelForCausalLM
                from smartloop_trainer.adapter_utils import load_adapters

                ws_model = self.workspace.get_model()
                self.logger.debug(f"Model directory: {ws_model.model_dir if ws_model else ''}")

                with torch.inference_mode():
                    slp_model = SLPModelForCausalLM(get_device_config(), self.logger)
                    _model, tokenizer = slp_model.load_for_workspace(self.workspace)
                    self.logger.debug(f"Model config: {_model.config.model_type if hasattr(_model.config, 'model_type') else 'Unknown'}")

                    project = self.workspace.current_project
                    if  not self.is_trainable:
                        self.logger.info("Loading base model with LoRA adapters...")
                        base_model = load_adapters(_model, project.project_dir, project.adapters if project else set(), self.logger, is_trainable=self.is_trainable)
                        model = PreTrainedLLM(model=base_model, tokenizer=tokenizer, logger=self.logger, project_id=self.workspace.current_project.id)
                    else:
                        model = PreTrainedLLM(model=_model, tokenizer=tokenizer, logger=self.logger, project_id=self.workspace.current_project.id)

                    return model
            else:
                ws_model = self.workspace.get_model()
                resolved_path = self.gguf_path or (ws_model.model_path if ws_model else "")
                if not os.path.exists(resolved_path):
                    raise FileNotFoundError(f"Model file not found: {resolved_path}")

                # Resolve LoRA adapter GGUF path (if any)
                lora_path = self._resolve_adapter_path()

                project = self.workspace.current_project

                # When the LLM worker pool is active, each worker holds its own
                # model copy on the same device.  Cap the main model's auto-sized
                # context so its KV cache doesn't claim the whole GPU and leave
                # room for the worker copies (avoids llama.cpp CUDA OOM crashes).
                main_max_n_ctx = None
                try:
                    from smartloop.config.app_settings import AppSettings
                    _s = AppSettings()
                    if _s.n_llm_workers and _s.n_llm_workers > 0:
                        main_max_n_ctx = _s.llm_main_n_ctx_cap
                        self.logger.info(
                            f"[model] Worker pool active ({_s.n_llm_workers}); "
                            f"capping main model context at {main_max_n_ctx}"
                        )
                except Exception:
                    pass

                model = LlmFactory.get_llama_cpp_instance(
                    resolved_path, logger=self.logger,
                    project_id=project.id if project else None,
                    lora_path=lora_path, max_n_ctx=main_max_n_ctx,
                )

                # print a table of model parameters and device configuration
                table = PrettyTable(align="l")
                table.field_names = ["Parameter", "Value"]
                table.add_row(["Project", next((p.id for p in self.workspace.projects if p.current), "N/A")])
                table.add_row(["Device", get_device_config().get_device_description()])
            
                print(table)
                
                return model

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e
