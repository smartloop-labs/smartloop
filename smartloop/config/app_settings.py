"""
Application settings configuration.
"""

import os
from typing import Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from smartloop.auth import credential_store     

class AppSettings(BaseSettings):
    """Application settings using Pydantic"""
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_encoding="utf-8",
        protected_namespaces=(),  # Disable protected namespaces to allow model_id field
        case_sensitive=False,
    )

    # LLM Configuration
    model_name: str = Field(default='gemma3-1b', validation_alias='slp_base_model_name')
    # Project ID
    project_id: Optional[str] = Field(default="d4c8f5a2-7b3e-4f91-a6d2-8e9c1b4a7f3d", validation_alias="slp_project_id")
    # Load quantized model
    load_quantized: bool = Field(default=False, validation_alias='slp_load_quantized')
    # Sentence Transformer
    sentence_transformer: str = Field(default='sentence-transformers/all-mpnet-base-v2', validation_alias='slp_sentence_transformer')
    # Hugging Face Token
    hf_token: str = Field(default='token', validation_alias='hf_token')
    # Chat History Configuration
    chat_history_token_size: int = Field(default=4096, validation_alias='slp_chat_history_token_size')
    # Section Token Configuration
    section_token_size: int = Field(default=2048, validation_alias='slp_section_token_size')
    # Logging
    log_level: str = Field(default='INFO', validation_alias='log_level')
    # Smartloop Platform API
    slp_api_base: str = Field(default='https://api.smartloop.ai', validation_alias='slp_api_base')
    # Developer Token for GCS downloads
    developer_token: Optional[str] = Field(default=None, validation_alias='slp_developer_token')

    # Version – set from smartloop.__version__ after init
    version: Optional[str] = Field(default=None)
    # Workspace root – set via SLP_HOME env var, defaults to ~/.smartloop
    slp_home: Optional[str] = Field(default=None, validation_alias='slp_home')

    # API Server Configuration
    api_host: str = Field(default="0.0.0.0", validation_alias="slp_api_host")
    api_port: int = Field(default=38540, validation_alias="slp_api_port")
    # Model Configuration
    preload_model: bool = Field(default=True, validation_alias="slp_api_preload_model")
    default_model: str = Field(default="gemma3-1b", validation_alias="slp_api_base_model_name")
    debug_mode: bool = Field(default=False, validation_alias="slp_api_debug_mode")
    # CORS Configuration
    cors_origins: str = Field(default="*", validation_alias="slp_api_cors_origins")
    # Parallel inference slots — allows multiple agents to run concurrently
    # in the same model instance via continuous batching (llama.cpp n_parallel).
    # Increases VRAM/RAM usage proportionally; keep at 1 on memory-constrained devices.
    n_parallel: int = Field(default=2, validation_alias="slp_n_parallel")
    # Number of LLM worker processes for parallel inference via
    # ProcessPoolExecutor.  Each worker loads its own model copy, so
    # RAM/VRAM usage scales linearly.  Defaults to n_parallel so the
    # number of workers matches the system's parallelism capacity.
    # Override via SLP_N_LLM_WORKERS if needed.
    n_llm_workers: Optional[int] = Field(
        default=None,
        validation_alias="slp_n_llm_workers",
        description="Number of LLM worker processes for parallel inference. Defaults to n_parallel.",
    )
    # Context window for each LLM worker's model copy.  Sized to hold a full
    # document's extraction input (slp_doc_extract_max_chars) plus room for the
    # output, so per-document extraction isn't truncated.  Larger = more VRAM
    # per worker copy (fewer fit on one GPU).
    llm_worker_n_ctx: int = Field(
        default=8192,
        validation_alias="slp_llm_worker_n_ctx",
        description="Context window per LLM worker model copy.",
    )
    # Cap applied to the MAIN model's (auto-sized) context window when the LLM
    # worker pool is active.  The main model's KV cache otherwise scales to the
    # whole device; capping it leaves room for the worker copies to share the
    # GPU.  Only caps (never raises) the auto-sized value.
    llm_main_n_ctx_cap: int = Field(
        default=4096,
        validation_alias="slp_llm_main_n_ctx_cap",
        description="Max context for the main model when the worker pool is active.",
    )
    # Cap applied to the large (agent) model's auto-sized context, and the
    # reserve kept free so loading it doesn't starve the main model + worker
    # pool.  The large model is loaded lazily on first agent use only if its
    # capped footprint fits in free VRAM after this reserve.
    llm_large_n_ctx_cap: int = Field(
        default=8192,
        validation_alias="slp_llm_large_n_ctx_cap",
        description="Max context for the large (agent) model when lazily loaded.",
    )
    llm_large_fit_reserve_gb: float = Field(
        default=1.0,
        validation_alias="slp_llm_large_fit_reserve_gb",
        description="VRAM (GB) kept free when fit-checking the large agent model.",
    )
    # Optional hard cap on characters fed to the per-document fan-out extractor.
    # 0 (default) = no fixed cap: the budget is derived from the extraction
    # model's actual context window (llm_worker_n_ctx for the worker pool, else
    # the main model's context), reserving room for the output — so a stored
    # document/chunk is processed without an arbitrary cutoff.  Set > 0 only to
    # force a smaller cap.
    doc_extract_max_chars: int = Field(
        default=0,
        validation_alias="slp_doc_extract_max_chars",
        description="Hard cap on fan-out extraction characters; 0 = derive from the extractor context.",
    )
    # Device for the sentence-transformer embedding model (chroma backend).
    # Defaults to CPU so it doesn't compete with the LLM copies for VRAM —
    # the GPU is reserved for the generative model(s).  Set to "cuda" only
    # when there's spare VRAM.  "auto" lets sentence-transformers choose.
    embedding_device: str = Field(
        default="cpu",
        validation_alias="slp_embedding_device",
        description='Device for the embedding model: "cpu" (default), "cuda", or "auto".',
    )
    # Sentence-transformer model for dense retrieval.  Defaults to the
    # MULTILINGUAL paraphrase model so a query in one language matches
    # documents in another (and paraphrases match).  Changing this requires
    # re-embedding stored documents — the sqlite store detects the change and
    # recomputes automatically.
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        validation_alias="slp_embedding_model",
        description="Sentence-transformer model name for dense retrieval.",
    )
    # Document parsing uses MarkItDown for fast text extraction and pushes OCR /
    # image understanding to a vision-capable model (selected by the "vision"
    # capability, like the orchestrator) served via the local /vision/v1
    # endpoint.  When enabled and a vision model is available, scanned/low-text
    # PDF pages are transcribed by that model to maximize extraction.
    vision_enabled: bool = Field(
        default=True,
        validation_alias="slp_vision_enabled",
        description="Use a vision-capable model for OCR/image understanding during document parsing.",
    )
    vision_ocr_min_chars_per_page: int = Field(
        default=100,
        validation_alias="slp_vision_ocr_min_chars_per_page",
        description="PDF pages with fewer extracted chars than this are treated as scanned and OCR'd by the vision model.",
    )
    # Optional smaller model for decision nodes (classify, match_skills).
    # Falls back to the main model when not set.
    decision_model: Optional[str] = Field(
        default=None, validation_alias="slp_decision_model",
        description="Small model for decision nodes. Falls back to main model.",
    )
    # Vector store backend for document/feedback/workspace retrieval.  Both do
    # dense (semantic relevancy) search — retrieval is fundamentally a
    # relevancy match, so dense handles paraphrases / different languages — with
    # a keyword-coverage boost for entity precision.
    #   "sqlite" — SQLite-stored embeddings + numpy cosine (default; no extra
    #              deps, falls back to TF-IDF only if the embedding model can't
    #              load).
    #   "chroma" — embedded ChromaDB (HNSW index; requires the chromadb
    #              package).
    vector_store_backend: str = Field(
        default="sqlite", validation_alias="slp_vector_store_backend",
        description='Vector store backend: "sqlite" or "chroma" (both dense).',
    )

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "AppSettings":
        """Resolve defaults that depend on other fields."""
        if self.developer_token is None:
            self.developer_token = credential_store.get_active_token()
        if self.n_llm_workers is None:
            self.n_llm_workers = self.n_parallel
        backend = (self.vector_store_backend or "sqlite").strip().lower()
        if backend not in ("sqlite", "chroma"):
            raise ValueError(
                f"Invalid vector_store_backend {self.vector_store_backend!r} — "
                'expected "sqlite" or "chroma".'
            )
        self.vector_store_backend = backend
        return self

    @property
    def home_dir(self) -> str:
        """Get the SLP home directory path.

        Uses the SLP_HOME env var when set (e.g. by a systemd service),
        otherwise defaults to ~/.smartloop.
        """
        if self.slp_home:
            return os.path.expanduser(self.slp_home)
        return os.path.expanduser("~/.smartloop")

    @property
    def bound_api_port(self) -> int:
        """Return the actually-bound API port.

        The server writes its real port to ``<home>/server.port`` after
        binding (the configured ``api_port`` may be 0 / auto).  Falls back to
        the configured port when the file is absent or unreadable.
        """
        try:
            port_file = os.path.join(self.home_dir, "server.port")
            with open(port_file) as f:
                return int(f.read().strip())
        except Exception:
            return self.api_port

    @property
    def vision_base_url(self) -> str:
        """OpenAI-compatible base URL for the local vision endpoint."""
        return f"http://127.0.0.1:{self.bound_api_port}/vision/v1"