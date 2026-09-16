"""
Application settings configuration.
"""

import os
from typing import Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from smartloop.auth import credential_store     
from smartloop.constants import BOOTSTRAP_MODEL, canonical_model_name

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
    model_name: str = Field(default=BOOTSTRAP_MODEL, validation_alias='slp_base_model_name')
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
    # Supported release channels (comma-separated). The first entry is the
    # default channel used when a client doesn't specify one, and the set
    # scopes which channels the /v1/releases endpoint will serve.
    release_channels_raw: str = Field(default='stable,beta', validation_alias='slp_release_channels')

    # Version – set from smartloop.__version__ after init
    version: Optional[str] = Field(default=None)
    # Workspace root – set via SLP_HOME env var, defaults to ~/.smartloop
    slp_home: Optional[str] = Field(default=None, validation_alias='slp_home')

    # API Server Configuration
    api_host: str = Field(default="0.0.0.0", validation_alias="slp_api_host")
    api_port: int = Field(default=38540, validation_alias="slp_api_port")
    # Model Configuration
    preload_model: bool = Field(default=True, validation_alias="slp_api_preload_model")
    default_model: str = Field(default=BOOTSTRAP_MODEL, validation_alias="slp_api_base_model_name")
    debug_mode: bool = Field(default=False, validation_alias="slp_api_debug_mode")
    # CORS Configuration
    cors_origins: str = Field(default="*", validation_alias="slp_api_cors_origins")
    # Parallel inference slots — allows multiple agents to run concurrently
    # in the same model instance via continuous batching (llama.cpp n_parallel).
    # Increases the KV cache proportionally.  Default 1 keeps VRAM free for the
    # on-demand skills/agent (large) model and the vision OCR model; raise it on
    # machines with VRAM headroom.  This is the only parallel-inference knob:
    # multi-process workers (each with its own model copy) were removed because
    # same-device copies serialize in the driver anyway.
    n_parallel: int = Field(default=1, validation_alias="slp_n_parallel")
    # Prompt-processing batch (llama.cpp n_batch).  Prefill is what a long
    # prompt costs — the whole tool catalog an MCP step is resolved against,
    # a document being summarized — and it runs one batch at a time, so 512
    # made a 3k-token prompt six passes when a GPU could do it in two.  Raised
    # on a GPU backend (see LlmFactory), where the buffer it grows is measured
    # in tens of MB; 0 keeps llama.cpp's own default.
    llm_n_batch: int = Field(
        default=0,
        validation_alias="slp_llm_n_batch",
        description="llama.cpp n_batch; 0 = choose from the device.",
    )
    # Generation threads (llama.cpp n_threads).  Only the layers that are not
    # offloaded run on them, so this matters most on a CPU-only load — where a
    # flat 4 left most of the machine idle.  0 = choose from the device.
    llm_n_threads: int = Field(
        default=0,
        validation_alias="slp_llm_n_threads",
        description="llama.cpp n_threads; 0 = choose from the device.",
    )
    # Context window for each LLM worker's model copy.  0 (default) = no cap:
    # Cap applied to the MAIN model's (auto-sized) context window.  Its KV
    # cache otherwise scales to the whole device, leaving nothing for the
    # on-demand large/vision models.  0 (default) = no cap — the main model
    # uses its full auto-sized window.  Only caps (never raises) it.
    llm_main_n_ctx_cap: int = Field(
        default=0,
        validation_alias="slp_llm_main_n_ctx_cap",
        description="Max context for the main model; 0 = no cap.",
    )
    # Cap applied to the large (agent) model's auto-sized context, and the
    # reserve kept free so loading it doesn't starve the main model.  The
    # large model is loaded lazily on first agent use only if its
    # capped footprint fits in free VRAM after this reserve.  0 (default) = no
    # cap — it uses its full auto-sized window.
    llm_large_n_ctx_cap: int = Field(
        default=0,
        validation_alias="slp_llm_large_n_ctx_cap",
        description="Max context for the large (agent) model when lazily loaded; 0 = no cap.",
    )
    llm_large_fit_reserve_gb: float = Field(
        default=1.0,
        validation_alias="slp_llm_large_fit_reserve_gb",
        description="VRAM (GB) kept free when fit-checking the large agent model.",
    )
    # Optional hard cap on characters fed to the per-document fan-out extractor.
    # 0 (default) = no fixed cap: the budget is derived from the extraction
    # model's actual context window, reserving room for the output — so a stored
    # document/chunk is processed without an arbitrary cutoff.  Set > 0 only to
    # force a smaller cap.
    doc_extract_max_chars: int = Field(
        default=0,
        validation_alias="slp_doc_extract_max_chars",
        description="Hard cap on fan-out extraction characters; 0 = derive from the extractor context.",
    )
    # Device/offload for the GGUF embedding model.  "auto" (default) offloads
    # fully to the GPU (Metal/Vulkan) — BGE-M3 is small (~400MB) and its encoder
    # is compute-heavy, so the GPU is worth it.  Set to "cpu" on memory-tight
    # devices to keep the GPU budget entirely for the generative model(s);
    # "cpu" runs the embedder with no GPU offload (n_gpu_layers=0).
    embedding_device: str = Field(
        default="auto",
        validation_alias="slp_embedding_device",
        description='Embedding device: "auto" (GPU offload, default) or "cpu".',
    )
    # GGUF embedding model for dense retrieval, run on the bundled llama.cpp
    # Metal/Vulkan backend (no torch).  Defaults to BGE-M3, which is MULTILINGUAL
    # so a query in one language matches documents in another (and paraphrases
    # match); Q4_K_M keeps the download/bundle small (~417MB).  Changing the
    # model changes the vector space, so the sqlite store detects the change via
    # the active embedding identity and re-embeds automatically.
    embedding_gguf_repo: str = Field(
        default="gpustack/bge-m3-GGUF",
        validation_alias="slp_embedding_gguf_repo",
        description="HF repo id of the GGUF embedding model (llama-cpp backend).",
    )
    embedding_gguf_file: str = Field(
        default="bge-m3-Q4_K_M.gguf",
        validation_alias="slp_embedding_gguf_file",
        description="GGUF filename within the repo.",
    )
    # Where the GGUF is fetched from on first run.  Our own public bucket is
    # tried before HuggingFace: the Hub gates downloads behind a token and is
    # slow from most regions, while this is an unauthenticated GCS object next
    # to the release archives.  Set to "" to skip it and go straight to HF.
    embedding_gguf_base_url: str = Field(
        default="https://dl.smartloop.ai/embeddings",
        validation_alias="slp_embedding_gguf_base_url",
        description=(
            "Base URL the embedding GGUF is downloaded from "
            "(<base>/<embedding_gguf_file>); empty falls back to HuggingFace."
        ),
    )
    # Document parsing uses MarkItDown for fast text extraction and pushes OCR /
    # image understanding to a vision-capable model (selected by the "vision"
    # capability, like the orchestrator) loaded in-process via get_vision_llm().
    # When enabled and a vision model is available, scanned/low-text PDF pages
    # are transcribed by that model to maximize extraction.
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
    vision_n_ctx: int = Field(
        default=8192,
        validation_alias="slp_vision_n_ctx",
        description="Context window for the vision OCR model. Must hold image tokens (~3-5k) plus the transcription output.",
    )
    vision_clip_device: str = Field(
        default="auto",
        validation_alias="slp_vision_clip_device",
        description='Device for the CLIP/mmproj image encoder: "auto" (GPU when enough VRAM is free, else CPU), "gpu" (force GPU — faster but OOM-segfaults if VRAM is tight), or "cpu" (always CPU).',
    )
    # Override for hybrid-reasoning models' <think> block (Qwen3.5 and friends,
    # see SLPModel.thinking_default).  Reasoning costs tokens and latency on
    # every call, enough that a long orchestration pass can run into
    # orchestrate_timeout_s, so slp_thinking_enabled=0 answers directly without
    # a code change (and =1 forces it on).  Unset (the default) uses each
    # model's own default; models with no reasoning mode are unaffected either
    # way.
    thinking_enabled: Optional[bool] = Field(
        default=None,
        validation_alias="slp_thinking_enabled",
        description="Override reasoning models' <think> block: 1 = think first (slower), 0 = answer directly. Unset uses each model's default.",
    )
    # Optional smaller model for decision nodes (classify, match_skills).
    # Falls back to the main model when not set.
    decision_model: Optional[str] = Field(
        default=None, validation_alias="slp_decision_model",
        description="Small model for decision nodes. Falls back to main model.",
    )
    # Wall-clock budget (seconds) for a single orchestration pass (planning +
    # tool/skill execution) before the streaming chat endpoint cancels it and
    # surfaces a timeout.  A native llama.cpp inference call can't be preempted
    # mid-token, so this bounds a stuck/runaway step (e.g. a small model that
    # never emits a stop token while executing several matched skills) instead
    # of hanging the client forever.  0 disables the watchdog.
    orchestrate_timeout_s: float = Field(
        default=180.0,
        validation_alias="slp_orchestrate_timeout_s",
        description="Max seconds for one orchestration pass before it is cancelled; 0 disables.",
    )
    # The same guard for the answer itself.  Orchestration is bounded above,
    # but the response stream that follows it used to wait on its producer
    # thread with no deadline at all: if that thread never produced a token —
    # wedged inside llama.cpp, or queued behind an earlier generation that is
    # (the answer runs on a single-worker pool) — the request hung until the
    # client gave up, with no error and no [DONE].  "It gets stuck and stays
    # stuck forever" is that path.
    #
    # Measured as *silence*, not total time: every chunk that arrives restarts
    # the clock, so a long answer never trips it however long it runs.  Larger
    # than orchestrate_timeout_s because the first token legitimately waits for
    # a model load and the prompt eval of a full context, which on a cold CPU
    # machine is minutes; only nothing-at-all for this long means stuck.
    # 0 disables the watchdog.
    stream_stall_timeout_s: float = Field(
        default=300.0,
        validation_alias="slp_stream_stall_timeout_s",
        description="Max seconds of silence from the response stream before it is abandoned; 0 disables.",
    )
    # --- per-project agent processes (see smartloop.supervisor) ------------
    # How long a project agent may sit unused before the supervisor stops it,
    # returning its baseline RSS.  A child carries FastAPI, the SQLite index
    # and the LangChain import graph even with no model in it, so ten open
    # projects is worth reclaiming.  0 keeps every started agent resident.
    agent_idle_ttl_s: float = Field(
        default=900.0,
        validation_alias="slp_agent_idle_ttl_s",
        description="Seconds a project agent may idle before it is stopped; 0 = never.",
    )
    # Resident-size ceiling for one project agent, in megabytes.  Separate
    # processes contain a *crash*; a ceiling contains a leak, which would
    # otherwise still take the machine down with it.  The offending child is
    # stopped and restarts on its project's next request; the others never
    # notice.  0 (default) = no ceiling.
    agent_memory_limit_mb: int = Field(
        default=0,
        validation_alias="slp_agent_memory_limit_mb",
        description="Per-project agent RSS ceiling in MB; 0 = no ceiling.",
    )

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "AppSettings":
        """Resolve defaults that depend on other fields."""
        if self.developer_token is None:
            self.developer_token = credential_store.get_active_token()
        # An existing .env may still name the orchestrator model by its old id;
        # normalize here so the rest of the framework only ever sees the
        # canonical name.
        self.model_name = canonical_model_name(self.model_name)
        self.default_model = canonical_model_name(self.default_model)
        return self

    @property
    def release_channels(self) -> list[str]:
        """Supported release channels; first entry is the default channel."""
        channels = [c.strip() for c in self.release_channels_raw.split(",") if c.strip()]
        return channels or ["stable"]

    @property
    def home_dir(self) -> str:
        """Get the SLP home directory path.

        Uses the SLP_HOME env var when set (e.g. by a systemd service),
        otherwise defaults to ~/.smartloop.
        """
        if self.slp_home:
            return os.path.expanduser(self.slp_home)
        return os.path.expanduser("~/.smartloop")