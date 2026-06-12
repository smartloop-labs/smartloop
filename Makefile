.PHONY: help venv build llama-wheel release-llama-wheel test test-unit test-cov build-binary sign notarize archive archive-zip publish-binary release-linux release-windows release clean clean-server clean-actions

UNAME_S := $(shell uname -s)
DIST_DIR := dist/slp
VENV := .venv

SLP_VERSION := $(shell grep '__version__' smartloop/__init__.py | sed 's/.*"\(.*\)"/\1/')
SLP_GCS_BUCKET := gs://smartloop-gcp-us-east-releases
DL_GCS_BUCKET := gs://dl.smartloop.ai
SERVER_PLATFORM := $(shell uname -s | tr '[:upper:]' '[:lower:]')
RAW_ARCH := $(shell uname -m)
ifeq ($(RAW_ARCH),x86_64)
    SERVER_ARCH := amd64
else
    SERVER_ARCH := $(RAW_ARCH)
endif
ARCHIVE_NAME := $(SERVER_PLATFORM)-$(SERVER_ARCH)-slp.tar.gz
DL_GCS_PATH := $(DL_GCS_BUCKET)/$(SLP_VERSION)

RELEASE_GCS_PATH := $(SLP_GCS_BUCKET)/slp/$(SLP_VERSION)
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

# Single source of truth for the llama-cpp-python version across all platforms.
LLAMA_VERSION := 0.3.16

# Portable CPU baseline for any from-source llama.cpp build.
#
# llama.cpp's CMake defaults GGML_NATIVE=ON (i.e. -march=native), which bakes
# the *build* machine's instruction set into the ggml-cpu backend with no
# runtime fallback. A binary compiled on a CI runner that has AVX-512 then
# faults with STATUS_ILLEGAL_INSTRUCTION (0xC000001D) on any CPU that lacks it
# — e.g. Intel Meteor Lake / Core Ultra, which has AVX2 but no AVX-512.
# Pin an explicit, portable x86-64-v3 (AVX2) baseline instead.
GGML_PORTABLE := -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF

# Prebuilt CUDA wheel from the abetlen index (Linux fallback; has our version).
LLAMA_ABETLEN_WHEEL = $(PIP) install llama-cpp-python==$(LLAMA_VERSION) --prefer-binary --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --extra-index-url=https://pypi.org/simple --force-reinstall --no-cache-dir

# Self-hosted portable llama-cpp-python wheels on GCS, served at
# dl.smartloop.ai/wheels. abetlen has no Windows CUDA wheel for our version (its
# Windows wheels stop at 0.3.4, too old for gemma3), and self-hosting lets us
# control the portable (GGML_NATIVE=OFF / AVX2) build per platform. Build and
# publish with `make release-llama-wheel` on the matching OS (the wheel's
# platform tag — win_amd64 / linux_x86_64 — comes from where you build it; a
# Windows wheel can only be built on Windows, a Linux wheel only on Linux).
LLAMA_WHEEL_GCS := $(SLP_GCS_BUCKET)/wheels
LLAMA_WHEELHOUSE := wheelhouse

ifeq ($(PLATFORM),windows)
    LLAMA_WHEEL_PLAT := win_amd64
    LLAMA_WHEEL_CMAKE := -DGGML_CUDA=ON $(GGML_PORTABLE)
else ifeq ($(PLATFORM),darwin)
    LLAMA_WHEEL_PLAT := macosx
    LLAMA_WHEEL_CMAKE := -DGGML_METAL=ON
else
    LLAMA_WHEEL_PLAT := linux_x86_64
    LLAMA_WHEEL_CMAKE := -DGGML_CUDA=ON $(GGML_PORTABLE)
endif
LLAMA_WHEEL_NAME := llama_cpp_python-$(LLAMA_VERSION)-cp311-cp311-$(LLAMA_WHEEL_PLAT).whl
LLAMA_WHEEL_URL := https://dl.smartloop.ai/wheels/$(LLAMA_WHEEL_NAME)

# From-source portable build fallback (needs the CUDA toolkit / nvcc on PATH).
LLAMA_SOURCE_BUILD = CMAKE_ARGS="$(LLAMA_WHEEL_CMAKE)" $(PIP) install llama-cpp-python==$(LLAMA_VERSION) --no-binary llama-cpp-python --index-url=https://pypi.org/simple --force-reinstall --no-cache-dir

# GPU-accelerated llama-cpp-python install: prefer our self-hosted GCS wheel,
# then fall back (abetlen wheel on Linux, from-source build elsewhere).
ifeq ($(PLATFORM),darwin)
    LLAMA_INSTALL_CMD = $(LLAMA_SOURCE_BUILD)
else ifeq ($(PLATFORM),linux)
    HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes || echo no)
    ifeq ($(CUDA),1)
        LLAMA_INSTALL_CMD = $(PIP) install "$(LLAMA_WHEEL_URL)" --force-reinstall --no-cache-dir || $(LLAMA_ABETLEN_WHEEL)
    else ifeq ($(HAS_NVIDIA),yes)
        LLAMA_INSTALL_CMD = $(PIP) install "$(LLAMA_WHEEL_URL)" --force-reinstall --no-cache-dir || $(LLAMA_ABETLEN_WHEEL)
    else
        LLAMA_INSTALL_CMD = CMAKE_ARGS="$(GGML_PORTABLE)" $(PIP) install llama-cpp-python==$(LLAMA_VERSION) --no-binary llama-cpp-python --index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
    endif
else
    LLAMA_INSTALL_CMD = $(PIP) install "$(LLAMA_WHEEL_URL)" --force-reinstall --no-cache-dir || $(LLAMA_SOURCE_BUILD)
endif


help:
	@echo "Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "  build           - Create venv, install requirements and llama-cpp-python with GPU support"
	@echo "  test            - Run slp --help and check llama-cpp GPU binding"
	@echo "  clean           - Remove .venv, build artifacts, and caches"
	@echo ""
	@echo "  Testing:"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-cov        - Run tests with coverage report"
	@echo ""
	@echo "  Packaging:"
	@echo "  build-binary    - Build slp binary with PyInstaller"
	@echo "  sign            - Code sign the binary (macOS only)"
	@echo "  notarize        - Notarize the binary with Apple (macOS only)"
	@echo "  archive         - Create slp tar.gz from dist/slp"
	@echo "  archive-zip     - Create windows-amd64-slp.zip from dist/slp"
	@echo "  llama-wheel        - Build the portable CUDA llama-cpp-python wheel (Windows)"
	@echo "  release-llama-wheel - Upload the prebuilt llama wheel to the release bucket"
	@echo "  publish-binary  - Build, sign, notarize, archive, and upload slp to GCS"
	@echo "  release-linux   - Build, archive, and upload linux-amd64-slp.tar.gz to the release bucket"
	@echo "  release-windows - Build, zip, and upload windows-amd64-slp.zip to the release bucket"
	@echo "  clean-server    - Remove PyInstaller build artifacts"
	@echo "  clean-actions   - Delete all GitHub Actions workflow runs"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	fi

build: venv
	@echo "=== Installing requirements ==="
	$(PIP) install -r requirements.txt --index-url=https://pypi.org/simple
	@echo "=== Installing llama-cpp-python with GPU support ==="
	$(LLAMA_INSTALL_CMD)
	@echo "=== Installing PyInstaller ==="
	$(PIP) install pyinstaller --index-url=https://pypi.org/simple
	@echo "=== Pre-downloading docling and EasyOCR models ==="
	$(PYTHON) -c "from docling.utils.model_downloader import download_models; download_models(with_easyocr=True)"
	@echo "=== Build complete ==="

# Build the portable CUDA llama-cpp-python wheel for the current OS. Requires the
# CUDA toolkit (nvcc) on PATH (Metal on macOS). GGML_NATIVE=OFF keeps the
# ggml-cpu backend at an AVX2 baseline so the wheel runs on any x86-64-v3 CPU,
# not just the builder's. Build Linux wheels on an old glibc (e.g. Ubuntu 22.04)
# so they load on older distros.
llama-wheel: venv
	@echo "=== Building portable llama-cpp-python $(LLAMA_VERSION) wheel for $(PLATFORM) ==="
	$(PIP) install --upgrade pip wheel --index-url=https://pypi.org/simple
	CMAKE_ARGS="$(LLAMA_WHEEL_CMAKE)" $(PIP) wheel llama-cpp-python==$(LLAMA_VERSION) --no-binary llama-cpp-python --no-deps --no-cache-dir --index-url=https://pypi.org/simple -w $(LLAMA_WHEELHOUSE)
	@ls -lh $(LLAMA_WHEELHOUSE)/$(LLAMA_WHEEL_NAME)

# Upload the prebuilt wheel to the release bucket (served at dl.smartloop.ai/wheels).
release-llama-wheel:
	@echo "=== Uploading $(LLAMA_WHEEL_NAME) to $(LLAMA_WHEEL_GCS) ==="
	gsutil cp $(LLAMA_WHEELHOUSE)/$(LLAMA_WHEEL_NAME) "$(LLAMA_WHEEL_GCS)/$(LLAMA_WHEEL_NAME)"
	gsutil ls -l "$(LLAMA_WHEEL_GCS)/"

test-deps:
	$(PIP) install -q -e ".[dev]"

test: test-deps
	@echo "=== Running tests ==="
	@echo "Checking slp --help..."
	@$(PYTHON) main.py --help
	@echo ""
	@echo "Checking llama-cpp-python GPU support..."
	@$(PYTHON) -c "from llama_cpp import llama_cpp; gpu=llama_cpp.llama_supports_gpu_offload(); print('GPU offload:', 'yes' if gpu else 'no (CPU-only)')"
	@echo ""
	$(PYTHON) -m pytest tests/unit -v
	$(PYTHON) -m pytest tests/api -v

test-unit: test-deps
	$(PYTHON) -m pytest tests/unit -v

test-cov: test-deps
	$(PYTHON) -m pytest tests/ --cov=smartloop --cov=commands --cov=tui --cov-report=term-missing

build-binary:
	@echo "Building slp v$(SLP_VERSION) with PyInstaller..."
	$(PYTHON) -m PyInstaller --noconfirm smartloop.spec
	@echo "Verifying binary..."
	@$(DIST_DIR)/slp --version || (echo "ERROR: slp binary verification failed" && exit 1)
	@echo "Binary verified successfully."
	@echo "Compiling .pyc files for faster startup..."
	@$(PYTHON) -m compileall -q -b $(DIST_DIR)

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
