# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import platform
import site
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# Collect all smartloop submodules, data files, and binaries
_tmp_ret = collect_all('smartloop')
datas += _tmp_ret[0]
binaries += _tmp_ret[1]
hiddenimports += _tmp_ret[2]

# Magika (MarkItDown's file-type detector) ships its ML model + config as
# package data that PyInstaller's analysis doesn't pick up. MarkItDown builds a
# Magika instance at construction time, so without the model dir it raises
# "model dir not found at .../magika/models/standard_v3_3" and every image/doc
# attachment fails before ever reaching the vision model.
_magika_ret = collect_all('magika')
datas += _magika_ret[0]
binaries += _magika_ret[1]
hiddenimports += _magika_ret[2]

# Bundle smartloop/skills (local package data — .md files etc.)
_skills_dir = Path('.') / 'smartloop' / 'skills'
if _skills_dir.exists():
    datas.append((str(_skills_dir), 'smartloop/skills'))

# Bundle the GGUF embedding model so a packaged install works offline with no
# first-run download.  Placed at <bundle>/embeddings/<file>, which
# smartloop.db.llama_embeddings._bundled_gguf_path() resolves via sys._MEIPASS.
# Pulled from our own public bucket rather than the HuggingFace Hub, which
# gates GGUF downloads behind a user token — the build machine should not need
# one.  Base URL/file mirror the app defaults (override via
# SLP_EMBEDDING_GGUF_* to match a customised default).  Best-effort: if the
# download fails the app still fetches it at runtime, so the build doesn't
# hard-fail on a transient outage.
try:
    import tempfile
    import urllib.request

    _emb_base = os.environ.get(
        'SLP_EMBEDDING_GGUF_BASE_URL', 'https://dl.smartloop.ai/embeddings'
    )
    _emb_file = os.environ.get('SLP_EMBEDDING_GGUF_FILE', 'bge-m3-Q4_K_M.gguf')
    _emb_url = f"{_emb_base.rstrip('/')}/{_emb_file}"
    _emb_gguf = str(Path(tempfile.gettempdir()) / _emb_file)
    if not Path(_emb_gguf).is_file():
        print(f"[spec] Downloading embedding model from {_emb_url}")
        urllib.request.urlretrieve(_emb_url, _emb_gguf + '.part')
        os.replace(_emb_gguf + '.part', _emb_gguf)
    datas.append((_emb_gguf, 'embeddings'))
    print(f"[spec] Bundled embedding model {_emb_url}")
except Exception as _emb_exc:
    print(
        f"[spec] WARNING: could not bundle embedding GGUF ({_emb_exc}); "
        "the app will download it at first use."
    )

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

# Bundle the Vulkan runtime so end users install nothing Vulkan-related.
# Only the GPU's own ICD/driver comes from the host (the same contract as
# libcuda.so.1 under the old CUDA build):
#  - Linux: bundle the loader (libvulkan.so.1) next to its only consumer,
#    libggml-vulkan.so in llama_cpp/lib — NOT the _internal root, which is the
#    dynamic loader's primary search dir and breaks numpy's OpenBLAS stack.
#  - Windows: PyInstaller skips System32 DLLs during dependency analysis, so
#    add the loader (vulkan-1.dll) explicitly. Machines whose GPU driver ships
#    its own copy in System32 resolve that one first; ours is the fallback.
#  - macOS: bundle MoltenVK (the Vulkan-on-Metal driver) plus an ICD manifest
#    rewritten to point at the bundled dylib; hooks/rthook_vulkan_icd.py aims
#    the loader at it. Metal remains available as the native backend.
if sys.platform.startswith('linux'):
    for _d in ['/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/usr/lib']:
        _p = Path(_d) / 'libvulkan.so.1'
        if _p.exists():
            binaries.append((str(_p), 'llama_cpp/lib'))
            break
elif sys.platform == 'win32':
    _vk_candidates = []
    if os.environ.get('VULKAN_SDK'):
        _vk_candidates.append(Path(os.environ['VULKAN_SDK']) / 'runtime' / 'x64' / 'vulkan-1.dll')
    _vk_candidates.append(Path(os.environ.get('WINDIR', r'C:\Windows')) / 'System32' / 'vulkan-1.dll')
    for _p in _vk_candidates:
        if _p.exists():
            binaries.append((str(_p), 'llama_cpp/lib'))
            break
elif sys.platform == 'darwin':
    import json
    # Homebrew's molten-vk formula moved the ICD manifest from share/ to etc/
    # (>= 1.4), so accept either location.
    _mvk_found = False
    for _prefix in [Path('/opt/homebrew'), Path('/usr/local')]:
        _mvk = _prefix / 'lib' / 'libMoltenVK.dylib'
        _icds = [
            _prefix / _sub / 'vulkan' / 'icd.d' / 'MoltenVK_icd.json'
            for _sub in ('etc', 'share')
        ]
        _icd = next((p for p in _icds if p.exists()), None)
        if _mvk.exists() and _icd is not None:
            binaries.append((str(_mvk), 'llama_cpp/lib'))
            _manifest = json.loads(_icd.read_text())
            _manifest['ICD']['library_path'] = './libMoltenVK.dylib'
            _out = Path('build') / 'MoltenVK_icd.json'
            _out.parent.mkdir(parents=True, exist_ok=True)
            _out.write_text(json.dumps(_manifest))
            datas.append((str(_out), 'llama_cpp/lib'))
            _mvk_found = True
            break
    if not _mvk_found:
        # Fail the build here rather than letting CI's post-build verify catch
        # a bundle that silently shipped without Vulkan support.
        raise SystemExit(
            'MoltenVK not found: need libMoltenVK.dylib plus MoltenVK_icd.json '
            'under /opt/homebrew or /usr/local (brew install molten-vk)'
        )

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

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'certifi',
        'version',
        'main',
        'llama_cpp',
        'llama_cpp.llama_cpp',
        'llama_cpp._ctypes_extensions',
        'pysqlite3',
        'sqlite3',
        'multipart',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/rthook_utf8.py', 'hooks/rthook_vulkan_icd.py'],
    excludes=[
        '.venv',
        'venv',
        # Inference runs entirely on llama.cpp (GGUF); torch/transformers are
        # only needed for the optional training extra (smartloop-trainer) and
        # must never land in the shipped bundle — torch alone is ~370MB and
        # drags the >1GB nvidia-cu12 CUDA runtime with it.
        'torch',
        'transformers',
        'accelerate',
        'sentence_transformers',
        'nvidia',
        'triton',
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

# Never strip. On Linux, numpy 2.x ships libscipy_openblas64_*.so built with a
# large ELF max-page-size; GNU strip on the build host rewrites it with
# non-page-aligned load segments, so the loader rejects it at runtime with
# "ELF load command address/offset not page-aligned" — which numpy rewraps as
# the misleading "import numpy from its source directory" ImportError. On macOS,
# stripping libtorch_cpu.dylib corrupts it enough that PyTorch's MPS Metal-shader
# JIT segfaults (SIGSEGV) at inference time — a crash seen only in the packaged
# build, never when running from the pristine venv wheels. Keep symbols intact.
_strip = False

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='slp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=_strip,
    upx=False,
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
    strip=_strip,
    upx=False,
    upx_exclude=[],
    name='slp',
)