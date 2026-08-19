.PHONY: help venv install build run run-debug run-local test test-unit test-cov build-binary check-apple-symbols sign notarize archive archive-zip publish-binary release-linux release-windows release clean clean-server clean-actions

UNAME_S := $(shell uname -s)
DIST_DIR := dist/slp
VENV := .venv

SLP_VERSION := $(shell grep '__version__' smartloop/__init__.py | sed 's/.*"\(.*\)"/\1/')
SLP_GCS_BUCKET := gs://smartloop-gcp-us-east-releases
DL_GCS_BUCKET := gs://dl.smartloop.ai
SERVER_PLATFORM := $(shell uname -s | tr '[:upper:]' '[:lower:]')
RAW_ARCH := $(shell uname -m)
LLAMA_VERSION := 0.3.34

ifeq ($(RAW_ARCH),x86_64)
    SERVER_ARCH := amd64
else
    SERVER_ARCH := $(RAW_ARCH)
endif
ARCHIVE_NAME := $(SERVER_PLATFORM)-$(SERVER_ARCH)-slp.tar.gz
DL_GCS_PATH := $(DL_GCS_BUCKET)/$(SLP_VERSION)

RELEASE_GCS_PATH := $(SLP_GCS_BUCKET)/slp/$(SLP_VERSION)

# The embedding GGUF is version-independent — it is named by model, immutable,
# and shared across releases — so it sits beside slp/ rather than under a
# version dir.  Served publicly at https://dl.smartloop.ai/embeddings/.
EMBED_GCS_PATH := $(SLP_GCS_BUCKET)/embeddings
EMBED_GGUF_FILE := $(shell grep -A 1 'embedding_gguf_file' smartloop/config/app_settings.py | grep 'default=' | sed 's/.*"\(.*\)".*/\1/')
WIN_ARCHIVE_NAME := windows-amd64-slp.zip

# Detect platform
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := darwin
else
    PLATFORM := windows
endif

# Set venv paths based on platform
ifeq ($(PLATFORM),windows)
    PIP := $(VENV)/Scripts/pip
    PYTHON := $(VENV)/Scripts/python
else
    PIP := $(VENV)/bin/pip
    PYTHON := $(VENV)/bin/python
endif

# ggml source builds default to -march=native; pin a portable x86-64-v3 (AVX2)
# baseline on Linux/Windows so CI-built binaries don't emit AVX-512 and SIGILL
# on CPUs that lack it (the failure the old prebuilt-wheel pin guarded against).
GGML_X86_PORTABLE := -DGGML_NATIVE=OFF -DGGML_AVX2=ON

# GPU backends: Metal + Vulkan (via MoltenVK) on macOS, Vulkan elsewhere.
# Every platform builds llama-cpp-python from source — there are no prebuilt
# Vulkan wheels — which requires the Vulkan SDK (headers + glslc) at build time.
ifeq ($(PLATFORM),darwin)
    LLAMA_CMAKE_ARGS := -DGGML_METAL=ON -DGGML_VULKAN=ON
else
    LLAMA_CMAKE_ARGS := -DGGML_VULKAN=ON $(GGML_X86_PORTABLE)
endif

# CMAKE_BUILD_PARALLEL_LEVEL parallelises the ninja/make invocation that pip
# drives internally. --find-links wheelhouse installs a compiled wheel cached
# by CI (actions/cache); when wheelhouse/ is absent or has no matching wheel it
# is a no-op and pip builds from the PyPI sdist with LLAMA_CMAKE_ARGS.
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
LLAMA_INSTALL_CMD = CMAKE_ARGS="$(LLAMA_CMAKE_ARGS)" CXXFLAGS="-I$(HOME)/.local/include" CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) $(PIP) install llama-cpp-python==$(LLAMA_VERSION) --find-links wheelhouse --index-url=https://pypi.org/simple --force-reinstall --no-cache-dir

# --- macOS: force the OpenBLAS SciPy wheel, not the Accelerate one ------------
# App Store review rejects any binary that references the deprecated Accelerate
# LAPACK symbols _dcabs1_ / _lsame_ / _xerbla_array__ ("The app uses or
# references the following non-public or deprecated APIs"), which blocks the Mac
# App Store build of Studio that bundles this framework.
#
# SciPy publishes two macOS arm64 wheels per release:
#   macosx_14_0_arm64  linked against /System/.../Accelerate.framework. Its
#                      scipy/linalg/cython_blas + cython_lapack import exactly
#                      those three legacy symbols — Accelerate's $NEWLAPACK
#                      interface has no replacement for them — and pip prefers
#                      this wheel on the macos-14 runner.
#   macosx_12_0_arm64  self-contained OpenBLAS in scipy/.dylibs. No Accelerate
#                      link, no reference to the flagged symbols.
# Identical SciPy version either way, so pin the OpenBLAS variant by wheel tag.
# SciPy is a transitive dep (scikit-learn), hence a post-install swap rather
# than a requirements.txt pin — there is no way to pin a wheel *variant* there.
# Verified by `make check-apple-symbols`, which CI runs after build-binary.
ifeq ($(PLATFORM)-$(RAW_ARCH),darwin-arm64)
SCIPY_OPENBLAS_CMD = set -e; \
	ver=$$($(PYTHON) -c 'import scipy; print(scipy.__version__)'); \
	py=$$($(PYTHON) -c 'import sys; print("%d%d" % sys.version_info[:2])'); \
	rm -rf build/scipy-openblas; \
	$(PIP) download "scipy==$$ver" --only-binary=:all: --no-deps \
		--platform macosx_12_0_arm64 --python-version $$py \
		--index-url=https://pypi.org/simple -d build/scipy-openblas; \
	$(PIP) install --force-reinstall --no-deps build/scipy-openblas/scipy-*.whl; \
	$(PYTHON) -c "import scipy.linalg.cython_blas, scipy.linalg.cython_lapack"
else
SCIPY_OPENBLAS_CMD = echo "  (not macOS arm64 — nothing to do)"
endif


help:
	@echo "Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "  install         - Create venv and install Python requirements"
	@echo "  build           - install + llama-cpp-python with GPU support + model downloads"
	@echo "  run             - Start the API server with default INFO logging"
	@echo "  run-debug       - Start the API server with LOG_LEVEL=DEBUG (verbose orchestrator/DAG/MCP logs)"
	@echo "  run-local       - Start the agent against the local API (http://localhost:8888)"
	@echo "  test            - Run slp --help and check llama-cpp GPU binding"
	@echo "  clean           - Remove .venv, build artifacts, and caches"
	@echo ""
	@echo "  Testing:"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-cov        - Run tests with coverage report"
	@echo ""
	@echo "  Packaging:"
	@echo "  build-binary    - Build slp binary with PyInstaller"
	@echo "  check-apple-symbols - Fail if dist/ references deprecated Accelerate LAPACK APIs (macOS)"
	@echo "  sign            - Code sign the binary (macOS only)"
	@echo "  notarize        - Notarize the binary with Apple (macOS only)"
	@echo "  archive         - Create slp tar.gz from dist/slp"
	@echo "  archive-zip     - Create windows-amd64-slp.zip from dist/slp"
	@echo "  publish-binary  - Build, sign, notarize, archive, and upload slp to GCS"
	@echo "  release-linux   - Build, archive, and upload linux-amd64-slp.tar.gz to the release bucket"
	@echo "  release-windows - Build, zip, and upload windows-amd64-slp.zip to the release bucket"
	@echo "  publish-embeddings - Upload the embedding GGUF (EMBED_GGUF=<path>) to the release bucket"
	@echo "  clean-server    - Remove PyInstaller build artifacts"
	@echo "  clean-actions   - Delete all GitHub Actions workflow runs"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	fi

install: venv
	@echo "=== Installing requirements ==="
	@# Inference runs on llama.cpp (GGUF); torch is NOT a runtime dependency and
	@# is intentionally absent from requirements.txt (it lives in the optional
	@# `training` extra). This keeps the shipped bundle ~370MB+ smaller and avoids
	@# pulling the >1GB nvidia-cu12 CUDA runtime that the default torch wheel drags.
	$(PIP) install -r requirements.txt --index-url=https://pypi.org/simple

build: install
	@echo "=== Installing llama-cpp-python with GPU support ==="
	$(LLAMA_INSTALL_CMD)
	@echo "=== Pinning numpy<2.4 (OpenBLAS ELF alignment crash with PyInstaller) ==="
	$(PIP) install "numpy>=1.24.0,<2.4" --force-reinstall --index-url=https://pypi.org/simple
	@echo "=== Pinning the OpenBLAS scipy wheel (no deprecated Accelerate LAPACK) ==="
	@$(SCIPY_OPENBLAS_CMD)
	@echo "=== Installing PyInstaller ==="
	$(PIP) install pyinstaller --index-url=https://pypi.org/simple
	@echo "=== Build complete ==="

run:
	$(PYTHON) -m main agent start

run-debug:
	LOG_LEVEL=DEBUG $(PYTHON) -m main agent start

run-local:
	SLP_API_BASE=http://localhost:8888 $(PYTHON) -m main agent start

test-deps:
	$(PIP) install -q -e ".[dev]"

test: test-deps
	@echo "=== Running tests ==="
	@echo "Checking slp --help..."
	@$(PYTHON) main.py --help
	@echo ""
	@echo "Checking llama-cpp-python GPU support..."
	@$(PYTHON) -c "from llama_cpp import llama_cpp; gpu=llama_cpp.llama_supports_gpu_offload(); print('GPU offload:', 'yes' if gpu else 'no (CPU-only)')" || echo "WARNING: llama-cpp-python GPU check crashed (binary incompatibility?)"
	@echo ""
	$(PYTHON) -m pytest tests/unit -v
	$(PYTHON) -m pytest tests/api -v

test-unit: test-deps
	$(PYTHON) -m pytest tests/unit -v

test-cov: test-deps
	$(PYTHON) -m pytest tests/ --cov=smartloop --cov-report=term-missing

build-binary:
	@echo "Building slp v$(SLP_VERSION) with PyInstaller..."
	$(PYTHON) -m PyInstaller --noconfirm smartloop.spec
	@echo "Verifying binary..."
	@$(DIST_DIR)/slp --version || (echo "ERROR: slp binary verification failed" && exit 1)
	@echo "Binary verified successfully."
	@echo "Compiling .pyc files for faster startup..."
	@$(PYTHON) -m compileall -q -b $(DIST_DIR)

# Guard against the App Store "non-public or deprecated APIs" rejection that
# blocks Studio's Mac App Store build, which bundles this framework wholesale.
# The offenders are the legacy Accelerate LAPACK helpers that have no
# $NEWLAPACK replacement, pulled in when a dependency links Accelerate rather
# than shipping its own BLAS (see SCIPY_OPENBLAS_CMD above). Scans every Mach-O
# in dist/ for an *undefined* reference — a library that defines the symbol
# itself, as bundled OpenBLAS does, is not an Apple API reference and is fine.
DEPRECATED_APPLE_SYMS := _dcabs1_|_lsame_|_xerbla_array__

check-apple-symbols:
ifeq ($(PLATFORM),darwin)
	@echo "=== Scanning $(DIST_DIR) for deprecated Accelerate LAPACK symbols ==="
	@hits=$$(find $(DIST_DIR) -type f \( -name "*.so" -o -name "*.dylib" \) -print0 \
		| xargs -0 -n1 sh -c 'nm -arch all "$$0" 2>/dev/null \
			| grep -qE "^ +U ($(DEPRECATED_APPLE_SYMS))$$" && echo "$$0"' \
		| sort); \
	if [ -n "$$hits" ]; then \
		echo "FAIL: these binaries reference deprecated Accelerate LAPACK APIs:"; \
		echo "$$hits" | sed 's/^/  /'; \
		echo "App Store review will reject the bundle. Rebuild the offending"; \
		echo "dependency against its own BLAS instead of Accelerate."; \
		exit 1; \
	fi
	@echo "OK: no references to deprecated Accelerate LAPACK APIs."
else
	@echo "check-apple-symbols: macOS only — skipped."
endif

sign:
ifeq ($(PLATFORM),darwin)
	@echo "=== Signing binary ==="
	@# Notarization rejects the bundle if ANY Mach-O inside lacks a Developer ID
	@# signature with hardened runtime, so detect binaries by content — matching
	@# by extension misses Python.framework/Versions/*/Python and torch/bin/*.
	@find $(DIST_DIR) -type f ! -path "$(DIST_DIR)/slp" -print0 | xargs -0 file | \
		grep 'Mach-O' | cut -d: -f1 | while read -r bin; do \
		codesign --force --options runtime --timestamp \
			--entitlements packaging/macos/entitlements.mac.plist \
			--sign "$(APPLE_DEVELOPER_ID)" "$$bin" || exit 1; \
	done
	codesign --force --options runtime --timestamp \
		--entitlements packaging/macos/entitlements.mac.plist \
		--sign "$(APPLE_DEVELOPER_ID)" \
		$(DIST_DIR)/slp
	@echo "Signing complete"
	codesign -v $(DIST_DIR)/slp
else
	@echo "Skipping code signing (not macOS)"
endif

notarize:
ifeq ($(PLATFORM),darwin)
	@echo "=== Notarizing binary ==="
	ditto -c -k --keepParent $(DIST_DIR) slp-notarize.zip
	@# notarytool exits 0 even on an Invalid verdict — check the status
	@# ourselves and surface Apple's per-file log before failing.
	@set -e; \
	xcrun notarytool submit slp-notarize.zip \
		--apple-id "$(APPLE_ID)" \
		--password "$(APPLE_APP_PASSWORD)" \
		--team-id "$(APPLE_TEAM_ID)" \
		--wait --output-format json | tee notarize-result.json; \
	ID=$$($(PYTHON) -c "import json; print(json.load(open('notarize-result.json'))['id'])"); \
	STATUS=$$($(PYTHON) -c "import json; print(json.load(open('notarize-result.json'))['status'])"); \
	if [ "$$STATUS" != "Accepted" ]; then \
		echo "Notarization failed with status: $$STATUS"; \
		xcrun notarytool log "$$ID" \
			--apple-id "$(APPLE_ID)" \
			--password "$(APPLE_APP_PASSWORD)" \
			--team-id "$(APPLE_TEAM_ID)"; \
		exit 1; \
	fi
	@rm -f slp-notarize.zip notarize-result.json
	@echo "Notarization complete"
else
	@echo "Skipping notarization (not macOS)"
endif

archive:
	@echo "Packaging $(ARCHIVE_NAME)..."
	@cd dist && tar czf ../$(ARCHIVE_NAME) slp/
	@echo "Created $(ARCHIVE_NAME)"

archive-zip:
	@echo "Packaging $(WIN_ARCHIVE_NAME)..."
	@$(PYTHON) -c "import shutil; shutil.make_archive('windows-amd64-slp', 'zip', 'dist', 'slp')"
	@echo "Created $(WIN_ARCHIVE_NAME)"

release-linux: build-binary archive
	@echo "Uploading $(ARCHIVE_NAME) to $(RELEASE_GCS_PATH)/linux-amd64-slp.tar.gz..."
	gsutil cp $(ARCHIVE_NAME) "$(RELEASE_GCS_PATH)/linux-amd64-slp.tar.gz"
	@rm -f $(ARCHIVE_NAME)
	@echo "Published linux build to $(RELEASE_GCS_PATH)/linux-amd64-slp.tar.gz"

release-windows: build-binary archive-zip
	@echo "Uploading $(WIN_ARCHIVE_NAME) to $(RELEASE_GCS_PATH)/windows-amd64-slp.zip..."
	gsutil cp $(WIN_ARCHIVE_NAME) "$(RELEASE_GCS_PATH)/windows-amd64-slp.zip"
	@rm -f $(WIN_ARCHIVE_NAME)
	@echo "Published windows build to $(RELEASE_GCS_PATH)/windows-amd64-slp.zip"

publish-binary: build-binary sign notarize archive
	@echo "Validating version..."
	@BINARY_VERSION=$$($(DIST_DIR)/slp --version 2>&1 | awk '{print $$NF}'); \
	if [ "$$BINARY_VERSION" != "$(SLP_VERSION)" ]; then \
		echo "ERROR: Binary version ($$BINARY_VERSION) does not match expected version ($(SLP_VERSION))"; \
		exit 1; \
	fi
	@echo "Version validated: $(SLP_VERSION)"
	@echo "Generating checksums..."
	@shasum -a 256 $(ARCHIVE_NAME) | sed 's| .*/| |' > checksums-sha256.txt
	@echo "Uploading to $(DL_GCS_PATH)/..."
	gsutil cp $(ARCHIVE_NAME) $(DL_GCS_PATH)/$(ARCHIVE_NAME)
	gsutil cp checksums-sha256.txt $(DL_GCS_PATH)/checksums-sha256.txt
	@rm -f $(ARCHIVE_NAME) checksums-sha256.txt
	@echo "Published to $(DL_GCS_PATH)/$(ARCHIVE_NAME)"

publish-embeddings:
	@if [ -z "$(EMBED_GGUF)" ]; then \
		echo "ERROR: set EMBED_GGUF=/path/to/$(EMBED_GGUF_FILE)"; \
		exit 1; \
	fi
	@echo "Generating checksum for $(EMBED_GGUF_FILE)..."
	@shasum -a 256 "$(EMBED_GGUF)" | sed 's| .*/| |' > checksums-sha256.txt
	@echo "Uploading to $(EMBED_GCS_PATH)/..."
	gsutil -h "Cache-Control:public, max-age=31536000, immutable" \
		cp "$(EMBED_GGUF)" $(EMBED_GCS_PATH)/$(EMBED_GGUF_FILE)
	gsutil -h "Cache-Control:public, max-age=300" \
		cp checksums-sha256.txt $(EMBED_GCS_PATH)/checksums-sha256.txt
	@rm -f checksums-sha256.txt
	@echo "Published to $(EMBED_GCS_PATH)/$(EMBED_GGUF_FILE)"

clean-server:
	@echo "Cleaning PyInstaller artifacts..."
	@rm -rf dist/slp build/slp slp-*.tar.gz
	@echo "Cleaned"

clean-actions:
	@echo "Cleaning Release workflow runs (keeping v$(SLP_VERSION))..."
	@gh run list --workflow "Release" --limit 100 --json databaseId,displayTitle --jq '.[] | select(.displayTitle != "v$(SLP_VERSION)") | "\(.databaseId)\t\(.displayTitle)"' | \
		while IFS=$$'\t' read -r id title; do \
			echo "Deleting: $$title ($$id)"; \
			gh run delete "$$id" 2>/dev/null || true; \
		done
	@echo "Done"

clean: clean-server
	@echo "Cleaning virtual environment..."
	@rm -rf $(VENV)
	@echo "Cleaning Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf *.egg-info
	@rm -rf build dist $(ARCHIVE_NAME)
	@echo "Cleaned"

.DEFAULT_GOAL := help
