from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
import hashlib
from typing import TYPE_CHECKING, Callable, Optional, Union
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from rich.console import Console
from prettytable import PrettyTable

from smartloop.utils import normalize_path
from smartloop.workspace.types import Document, DocumentStatus, Skill
from smartloop.loader import ModelLoader
from smartloop.model.slp_model import SLPModel
from smartloop.model.downloader import SLPDownloader
from smartloop.llms.base_llm import BaseLLM
from smartloop.paths import expand_home

if TYPE_CHECKING:
    from smartloop.workspace import Workspace
    
console = Console()

class GenericModel(SLPModel):
    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, logger)
        self.downloader = downloader
        self._instance_cache = None  # Cache for loaded model

    def init(self):
        self.logger.info("[info] Initializing model...")
        self.logger.info("=" * 50)

        self.downloader.download(model_id=self.model_id())

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hasher = hashlib.md5()

        with open(file_path, 'rb') as f:
            hasher.update(f.read())

        return hasher.hexdigest()

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL"""
        try:
            result = urlparse(str(source))
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_google_drive_url(self, url: str) -> bool:
        """Check if URL is a Google Drive link"""
        return 'drive.google.com' in url
    
    def _extract_google_drive_id(self, url: str) -> str:
        """Extract file ID from Google Drive URL"""
        # Handle different Google Drive URL formats
        # Format 1: https://drive.google.com/file/d/FILE_ID/view
        # Format 2: https://drive.google.com/open?id=FILE_ID
        
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract file ID from Google Drive URL: {url}")
    
    def _copy_to_project_dir(self, file_path: Path, dest_dir: Path) -> Path:
        """Copy the original file into the project directory, handling name collisions."""
        import shutil
        os.makedirs(dest_dir, exist_ok=True)
        dest = dest_dir / file_path.name
        # Handle name collisions by appending a counter
        if dest.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(str(file_path), str(dest))
        self.logger.info(f"Copied to project dir: {dest.name}")
        return dest

    def _generate_filename(self, file_path: Path, llm: "BaseLLM") -> Optional[str]:
        """Use the LLM to produce a short descriptive filename from file content."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")[:512]
        except Exception:
            return None

        if not content.strip():
            return None

        try:
            prompt = (
                "Generate a short, descriptive filename (2-5 words, snake_case, no extension) "
                "for a document that starts with:\n\n"
                f"{content}\n\n"
                "Reply with ONLY the filename, nothing else."
            )
            raw = llm.invoke(prompt)
            name = (raw.text if hasattr(raw, "text") else str(raw)).strip().strip('"\'')
            name = Path(name).stem
            name = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")[:60]
            if name:
                self.logger.info(f"[add_document] Generated filename: {name}")
                return name
        except Exception as e:
            self.logger.warning(f"[add_document] LLM filename generation failed: {e}")

        return None

    def _download_from_google_drive(self, url: str) -> Path:
        """Download file from Google Drive to a temp directory"""
        try:
            tmp_dir = Path(tempfile.mkdtemp())

            file_id = self._extract_google_drive_id(url)
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            self.logger.info(f"Downloading from Google Drive: {file_id}")

            client = httpx.Client(follow_redirects=True, timeout=60)
            response = client.get(download_url)

            # Check if we need to confirm download for large files
            confirm_token = None
            for name, value in response.cookies.items():
                if name.startswith("download_warning"):
                    confirm_token = value
                    break

            if confirm_token:
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                response = client.get(download_url)

            filename = None
            if "Content-Disposition" in response.headers:
                content_disp = response.headers["Content-Disposition"]
                match = re.search(r'filename="?(.+)"?', content_disp)
                if match:
                    filename = match.group(1).strip('"')

            if not filename:
                filename = f"gdrive_{file_id}.pdf"

            download_path = tmp_dir / filename

            with open(download_path, "wb") as f:
                f.write(response.content)

            self.logger.info(f"Downloaded to: {download_path}")

            return download_path

        except Exception as e:
            self.logger.error(f"Failed to download from Google Drive {url}: {e}")
            raise

    def _download_url(self, url: str) -> Path:
        """Download file from URL to a temp directory"""
        try:
            tmp_dir = Path(tempfile.mkdtemp())

            # Get filename from URL or generate one
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or f"download_{hashlib.md5(url.encode()).hexdigest()[:8]}"

            download_path = tmp_dir / filename

            self.logger.info(f"Downloading from URL: {url}")

            response = httpx.get(url, follow_redirects=True, timeout=60)
            response.raise_for_status()

            with open(download_path, "wb") as out_file:
                out_file.write(response.content)

            self.logger.info(f"Downloaded to: {download_path}")

            return download_path
        except Exception as e:
            self.logger.error(f"Failed to download from URL {url}: {e}")
            raise

    def _emit_progress(self, callback: Optional[Callable[[dict], None]], stage: str, filename: str, message: str):
        """Emit a progress event if a callback is provided."""
        if callback:
            callback({"stage": stage, "filename": filename, "message": message})

    def add_document(self, source: Union[str, Path], progress_callback: Optional[Callable[[dict], None]] = None, cwd: str | None = None, generate_filename: bool = False) -> list[Document]:
        """Add a document file, directory, or URL to the project.

        Copies the original file into the project directory and registers it
        with PENDING status.  The indexer handles conversion, indexing, and
        summary generation in the background.
        """
        console.print(f"\n[i] Adding files to be processed:")

        current_project = self.workspace.current_project
        if not current_project:
            self.logger.error("[error] No current project found.")
            return []

        # Documents are project state, so they always land under project_dir —
        # never under a caller-supplied cwd.  ``cwd`` is the user's own folder
        # for file/bash tools; writing ingested copies there put project state
        # outside the framework's home, where it could not follow SLP_HOME.
        dest_dir = Path(current_project.documents_dir)

        documents: list[Document] = []

        # Handle URL
        if isinstance(source, str) and self._is_url(source):
            try:
                self._emit_progress(progress_callback, "downloading", source, f"Downloading from URL...")
                if self._is_google_drive_url(source):
                    downloaded_path = self._download_from_google_drive(source)
                else:
                    downloaded_path = self._download_url(source)

                self._emit_progress(progress_callback, "registering", downloaded_path.name, f"Registering {downloaded_path.name}...")
                dest_path = self._copy_to_project_dir(downloaded_path, dest_dir)

                documents.append(Document(
                    id=self._calculate_file_hash(downloaded_path),
                    path=str(dest_path),
                    name=downloaded_path.name,
                    summary=None,
                    status=DocumentStatus.PENDING
                ))

                # Clean up temp download directory
                import shutil
                tmp_dir = downloaded_path.parent
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                self.logger.error(f"[error] Failed to download URL: {source}")
                return []
        else:
            # Normalize pasted file paths
            if isinstance(source, str):
                source = source.strip().strip("'\"")
                source = source.replace("\\ ", " ")
                if source.lower().startswith("file://"):
                    source = source[7:]
                source = normalize_path(source)
            source_path = Path(expand_home(source)) if isinstance(source, str) else source

            if source_path.is_dir():
                for file in os.listdir(source_path):
                    file_path = Path(source_path) / file
                    if file_path.is_file():
                        try:
                            self._emit_progress(progress_callback, "registering", file_path.name, f"Registering {file_path.name}...")
                            dest_path = self._copy_to_project_dir(file_path, dest_dir)
                            documents.append(Document(
                                id=self._calculate_file_hash(file_path),
                                path=str(dest_path),
                                name=file_path.name,
                                summary=None,
                                status=DocumentStatus.PENDING
                            ))
                        except Exception as e:
                            self.logger.error(f"Failed to register {file_path.name}: {e}")
            elif source_path.is_file():
                try:
                    self._emit_progress(progress_callback, "registering", source_path.name, f"Registering {source_path.name}...")
                    dest_path = self._copy_to_project_dir(source_path, dest_dir)
                    documents.append(Document(
                        id=self._calculate_file_hash(source_path),
                        path=str(dest_path),
                        name=source_path.name,
                        summary=None,
                        status=DocumentStatus.PENDING
                    ))
                except Exception as e:
                    self.logger.error(f"[error] Failed to register file: {source_path}: {e}")
                    raise
            else:
                raise FileNotFoundError(f"Invalid file path: {source_path}")

        existing_ids = {doc.id for doc in current_project.documents}
        new_docs = [doc for doc in documents if doc.id not in existing_ids]
        skipped = len(documents) - len(new_docs)
        if skipped:
            self.logger.info(f"[add_document] {skipped} document(s) skipped (duplicate hash)")

        if generate_filename and new_docs:
            try:
                llm = self.get_instance()
                batch_paths = {Path(d.path).resolve() for d in new_docs}
                for doc in new_docs:
                    doc_path = Path(doc.path)
                    if doc_path.suffix.lower() != ".txt":
                        continue
                    generated = self._generate_filename(doc_path, llm)
                    if not generated:
                        continue
                    new_path = doc_path.parent / f"{generated}.txt"
                    if new_path.resolve() != doc_path.resolve() and new_path.exists() and new_path.resolve() not in batch_paths:
                        counter = 1
                        while new_path.exists() and new_path.resolve() not in batch_paths:
                            new_path = doc_path.parent / f"{generated}_{counter}.txt"
                            counter += 1
                    batch_paths.discard(doc_path.resolve())
                    batch_paths.add(new_path.resolve())
                    doc_path.rename(new_path)
                    doc.path = str(new_path)
                    self.logger.info(f"[add_document] Renamed {doc_path.name} → {new_path.name}")
            except Exception as e:
                self.logger.warning(f"[add_document] Filename generation failed: {e}")

        for doc in new_docs:
            doc_name = Path(doc.path).name
            console.print(f"\t[green] added: {doc_name}[/green] ")

        current_project.documents.extend(new_docs)
        self.logger.info(
            f"[add_document] project={current_project.id}, dest_dir={dest_dir}, "
            f"{len(new_docs)} new, {len(current_project.documents)} total documents"
        )
        return new_docs
    
    def add_skill(self, name: str, content: str):
        """Add a skill to the current project."""
        current_project = self.workspace.current_project
        if not current_project:
            self.logger.error("[error] No current project found.")
            return

        content = content.strip()
        if not content:
            self.logger.error("[error] Skill content cannot be empty.")
            return

        if len(content) < 10:
            self.logger.warning("[warn] Skill content is very short. Consider adding more details.")

        skill = Skill(id=str(uuid4())[:8], name=name, content=content)
        current_project.skills.append(skill)

        self.logger.info(f"[ok] Skill '{name}' added")
        self.logger.info(f"Total skills: {len(current_project.skills)}")

    def status(self):
        self.logger.info("[info] Project Status:")
        console.print()

        current_project = self.workspace.current_project
        if not current_project:
            self.logger.error("[error] No project found.")
            return
        
        table = PrettyTable(align="l")
        table.field_names = ["File", "Status"]

        to_be_processed = 0
      
        for doc in current_project.documents:
            if doc.status == DocumentStatus.PENDING:
                if Path(doc.path).exists():
                    table.add_row([Path(doc.path).name, DocumentStatus.PENDING.name])
                    to_be_processed += 1
            elif doc.status == DocumentStatus.INDEXED:
                table.add_row([Path(doc.path).name, DocumentStatus.INDEXED.name])
            elif doc.status == DocumentStatus.DELETED:
                    table.add_row([Path(doc.path).name, DocumentStatus.DELETED.name])
            else:
                table.add_row([Path(doc.path).name, DocumentStatus(doc.status).name])

            print(table)

        if to_be_processed == 0:
            self.logger.info("\n- No files to process.")
        else:
            self.logger.info(f"\n({to_be_processed}) files to process.")

    def _check_safetensors(self, model_dir: str) -> None:
        """Raise if no safetensors files exist in the model directory."""
        if not os.path.isdir(model_dir) or not any(
            f.endswith(".safetensors") for f in os.listdir(model_dir)
        ):
            raise FileNotFoundError(
                f"No safetensors files found in {model_dir}. "
                "Training and building require the full model weights. "
                "Download them using an HF token: slp init --hf-token <token>"
            )

    def train(self, progress_callback=None):
        from smartloop.trainer import Trainer

        self.logger.info("[info] Training model...")
        self.logger.info("=" * 50)

        project = self.workspace.current_project
        if not project:
            raise ValueError("No active project. Please create a project first.")

        ws_model = self.workspace.get_model()
        model_id = ws_model.model_id if ws_model else ""
        model_dir = self.workspace.model_dir_for(model_id)
        self._check_safetensors(model_dir)

        trainer = Trainer(self.workspace, self.logger)
        trainer.train(progress_callback=progress_callback)

    def build(self, progress_callback=None):
        from smartloop.trainer import Builder

        self.logger.info("[info] Building model...")
        self.logger.info("=" * 50)

        ws_model = self.workspace.get_model()
        model_id = ws_model.model_id if ws_model else ""

        model_dir = self.workspace.model_dir_for(model_id)
        self._check_safetensors(model_dir)

        os.makedirs(ws_model.bin_dir, exist_ok=True)
        quantize_type = ws_model.config.quantize if ws_model.config else "q8_0"
        builder = Builder(self.workspace, model_id, self.logger, output_dir=ws_model.bin_dir)
        builder.build(quantize_type=quantize_type, progress_callback=progress_callback)

    def get_instance(self) -> BaseLLM:
        if self._instance_cache is None:
            model_id = self.model_id()
            # Cascade: use project-level GGUF if it exists, else fall back to workspace GGUF
            gguf_path = self.workspace.resolve_gguf_path(model_id)

            self.logger.info(f"Loading model instance from GGUF: {gguf_path}")
            model_loader = ModelLoader(self.workspace, self.logger, gguf_path=gguf_path)
            self._instance_cache = model_loader.load()

        return self._instance_cache

