# -*- mode: python ; coding: utf-8 -*-
import sys
import platform
import site
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata, collect_all

datas = []
binaries = []
hiddenimports = []

# Collect all smartloop submodules, data files, and binaries
_tmp_ret = collect_all('smartloop')
datas += _tmp_ret[0]
binaries += _tmp_ret[1]
hiddenimports += _tmp_ret[2]

# Bundle smartloop/skills (local package data — .md files etc.)
_skills_dir = Path('.') / 'smartloop' / 'skills'
if _skills_dir.exists():
    datas.append((str(_skills_dir), 'smartloop/skills'))

# Include package metadata required at runtime (importlib.metadata lookups)
for pkg in ['docling-slim', 'docling-core', 'docling-parse']:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Include docling_parse resource directories (fonts, encodings, glyphs for PDF parsing)
for sp in site.getsitepackages():
    dp_dir = Path(sp) / 'docling_parse'
    if dp_dir.exists():
        for res_name in ['pdf_resources', 'pdf_resources_v2']:
            res_dir = dp_dir / res_name
            if res_dir.exists():
                datas.append((str(res_dir), f'docling_parse/{res_name}'))
        break

# Include docling_core resource schemas (JSON document format schemas)
for sp in site.getsitepackages():
    schemas_dir = Path(sp) / 'docling_core' / 'resources'
    if schemas_dir.exists():
        datas.append((str(schemas_dir), 'docling_core/resources'))
        break


# Collect tui css files
tui_css_dir = Path('.') / 'tui' / 'css'
if tui_css_dir.exists():
    datas.append((str(tui_css_dir), 'tui/css'))

# Collect llama_cpp shared libraries (libllama, libggml, etc.)
# These are required by the smartloop framework for model inference.
for sp in site.getsitepackages():
    llama_lib_dir = Path(sp) / 'llama_cpp' / 'lib'
    if llama_lib_dir.exists():
        for lib_file in llama_lib_dir.glob('*.so*' if sys.platform != 'win32' else '*.dll'):
            binaries.append((str(lib_file), 'llama_cpp/lib'))
        for lib_file in llama_lib_dir.glob('*.dylib'):
            binaries.append((str(lib_file), 'llama_cpp/lib'))
        break

# Bundle CUDA runtime libraries so the binary runs on machines that have the
# NVIDIA driver but not the full CUDA toolkit. libcuda.so.1 is intentionally
# excluded — it is the driver stub and must come from the host driver.
if sys.platform != 'win32':
    _cuda_runtime_libs = [
        'libcudart.so.12',
        'libcublas.so.12',
        'libcublasLt.so.12',
    ]
    import ctypes.util, subprocess
    _cuda_search_dirs = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12/lib64',
    ]
    for _lib in _cuda_runtime_libs:
        for _d in _cuda_search_dirs:
            _p = Path(_d) / _lib
            if _p.exists():
                binaries.append((str(_p), '.'))
                break

try:
    import chromadb
    chromadb_dir = Path(chromadb.__file__).parent
    datas.append((str(chromadb_dir), 'chromadb'))
except ImportError:
    pass

# Collect rich unicode data (dynamically imported via importlib, filenames contain hyphens)
try:
    import rich._unicode_data as _rud
    rud_dir = Path(_rud.__file__).parent
    datas.append((str(rud_dir), 'rich/_unicode_data'))
except ImportError:
    pass

# Collect huggingface_hub templates (required for model card generation during save_pretrained)
try:
    import huggingface_hub
    hf_hub_dir = Path(huggingface_hub.__file__).parent
    templates_dir = hf_hub_dir / 'templates'
    if templates_dir.exists():
        datas.append((str(templates_dir), 'huggingface_hub/templates'))
except ImportError:
    pass

# Collect python-docx templates (required for header/footer processing in Word documents)
# Also include docx/parts directory so that relative path resolution
# (e.g. docx/parts/../templates/default-header.xml) works at runtime.
try:
    import docx
    docx_dir = Path(docx.__file__).parent
    templates_dir = docx_dir / 'templates'
    if templates_dir.exists():
        datas.append((str(templates_dir), 'docx/templates'))
    parts_init = docx_dir / 'parts' / '__init__.py'
    if parts_init.exists():
        datas.append((str(parts_init), 'docx/parts'))
except ImportError:
    pass

# Collect EasyOCR models and data (OCR engine for docling)
try:
    import easyocr
    easyocr_dir = Path(easyocr.__file__).parent
    datas.append((str(easyocr_dir), 'easyocr'))
except ImportError:
    pass


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'certifi',
        'version',
        'main',
        'tui',
        'tui.chat',
        'textual',
        'prompt_toolkit',
        'llama_cpp',
        'llama_cpp.llama_cpp',
        'llama_cpp._ctypes_extensions',
        'docling',
        'docling.document_converter',
        'docling.models.plugins',
        'docling.models.plugins.defaults',
        'docling.pipeline.base_pipeline',
        'easyocr',
        'chromadb',
        'chromadb.config',
        'chromadb.api',
        'chromadb.db',
        'chromadb.db.impl',
        'chromadb.db.impl.sqlite',
        'chromadb.segment',
        'chromadb.telemetry',
        'chromadb.telemetry.product.posthog',
        'chromadb.api.rust',
        'chromadb_rust_bindings',
        'posthog',
        'pysqlite3',
        'sqlite3',
        'multipart',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/rthook_utf8.py', 'hooks/rthook_chroma_telemetry.py'],
    excludes=[
        '.venv',
        'venv',
        'tensorflow',
        'tf_keras',
        'keras',
        'smartloop_trainer',
        'peft',
        'trl',
        'datasets',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='slp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=sys.platform != 'win32',
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=platform.machine() if sys.platform == 'darwin' else None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=sys.platform != 'win32',
    upx=True,
    upx_exclude=[],
    name='slp',
)