import os
from pathlib import Path

from .base_parser import BaseParser
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, HTMLFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, AcceleratorOptions
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat


def _accelerator_device() -> AcceleratorDevice:
    """Pick the accelerator device for docling's layout / OCR models.

    Default preference: MPS → CUDA → CPU.  The ``slp_docling_device`` setting
    can override (e.g. "cpu" to avoid GPU contention with llama.cpp).
    """
    try:
        from smartloop.config.app_settings import AppSettings
        pref = (AppSettings().docling_device or "auto").strip().lower()
    except Exception:
        pref = "auto"

    if pref == "cpu":
        return AcceleratorDevice.CPU
    if pref == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return AcceleratorDevice.CUDA
        except ImportError:
            pass
        return AcceleratorDevice.CPU

    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return AcceleratorDevice.MPS
        if torch.cuda.is_available():
            return AcceleratorDevice.CUDA
    except ImportError:
        pass
    return AcceleratorDevice.CPU


def _easyocr_model_dir() -> str:
    """Return the local EasyOCR model directory, preferring the bundled copy."""
    # 1. Docling's pre-downloaded cache
    docling_easyocr = os.path.join(str(Path.home()), ".cache", "docling", "models", "EasyOcr")
    if os.path.isdir(docling_easyocr) and os.listdir(docling_easyocr):
        return docling_easyocr
    # 2. PyInstaller bundle or dev install: models alongside the easyocr package
    try:
        import easyocr
        pkg_models = os.path.join(os.path.dirname(easyocr.__file__), "model")
        if os.path.isdir(pkg_models) and os.listdir(pkg_models):
            return pkg_models
    except ImportError:
        pass
    # 3. Default user-level cache
    default = os.path.join(str(Path.home()), ".EasyOCR", "model")
    os.makedirs(default, exist_ok=True)
    return default


def _docling_artifacts_path():
    """Return local docling model artifacts path if models are cached."""
    try:
        from docling.datamodel.settings import settings as docling_settings
        cache = docling_settings.cache_dir / "models"
        if cache.is_dir() and any(cache.iterdir()):
            return cache
    except Exception:
        pass
    return None


class DocumentParser(BaseParser):
    def _build_converter(self, device: AcceleratorDevice) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(device=device)
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(
            force_full_page_ocr=True,
            model_storage_directory=_easyocr_model_dir(),
            download_enabled=False,
        )

        artifacts = _docling_artifacts_path()
        if artifacts is not None:
            pipeline_options.artifacts_path = artifacts

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.DOCX: WordFormatOption(),
                InputFormat.HTML: HTMLFormatOption(),
            }
        )

    def parse(self) -> str:
        """Convert the document to markdown using DocumentConverter."""
        import logging
        logger = logging.getLogger(__name__)

        device = _accelerator_device()
        converter = self._build_converter(device)

        from smartloop.utils.torch_meta_guard import safe_model_load
        try:
            with safe_model_load():
                result = converter.convert(self.file_path)
        except Exception:
            if device != AcceleratorDevice.CPU:
                logger.warning(f"Docling failed on {device.value}, falling back to CPU")
                converter = self._build_converter(AcceleratorDevice.CPU)
                with safe_model_load():
                    result = converter.convert(self.file_path)
            else:
                raise

        return str(result.document.export_to_markdown())
