.PHONY: help venv test build sign notarize release clean ensure-uv dist archive llama-wheel publish-wheel

VENV := .venv

# Detect platform: linux, darwin, or windows
UNAME_S := $(shell uname -s)
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

# Detect architecture: arm64 or amd64
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    ARCH := arm64
else ifeq ($(UNAME_M),aarch64)
    ARCH := arm64
else
    ARCH := amd64
endif

# Read version from installed smartloop package
VERSION := $(shell $(PYTHON) -c "from importlib.metadata import version; print(version('slp'))" 2>/dev/null)

help:
	@echo "Available targets:"
	@echo "  build        - Set up the local dev environment (editable install in .venv)"
	@echo "  test         - Smoke-test the dev environment"
	@echo "  dist         - Build the relocatable standalone distribution (dist/slp-app)"
	@echo "  archive      - dist + tar -> <platform>-<arch>-slp.tar.gz"
	@echo "  llama-wheel  - Build the portable GPU llama-cpp-python wheel to publish"
	@echo "  sign         - Code sign the distribution (macOS only)"
	@echo "  notarize     - Notarize the distribution with Apple (macOS only)"
	@echo "  release      - Full release: dist + sign + notarize + archive"
	@echo "  clean        - Clean all build artifacts"

# Detect GPU backend for llama-cpp-python
ifeq ($(PLATFORM),darwin)
    LLAMA_CMAKE_ARGS := -DGGML_METAL=ON
    LLAMA_INSTALL_CMD = CMAKE_ARGS="$(LLAMA_CMAKE_ARGS)" $(PIP) install llama-cpp-python --index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
else ifeq ($(PLATFORM),linux)
    HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes || echo no)
    ifeq ($(CUDA),1)
        LLAMA_INSTALL_CMD = $(PIP) install llama-cpp-python --prefer-binary --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --extra-index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
    else ifeq ($(HAS_NVIDIA),yes)
        LLAMA_INSTALL_CMD = $(PIP) install llama-cpp-python --prefer-binary --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --extra-index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
    else
        # No NVIDIA GPU — fall back to CPU-only build
        LLAMA_INSTALL_CMD = $(PIP) install llama-cpp-python --index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
    endif
else ifeq ($(PLATFORM),windows)
    HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes || echo no)
    ifeq ($(CUDA),1)
        LLAMA_INSTALL_CMD = $(PIP) install llama-cpp-python --prefer-binary --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --extra-index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
    else ifeq ($(HAS_NVIDIA),yes)
        LLAMA_INSTALL_CMD = $(PIP) install llama-cpp-python --prefer-binary --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --extra-index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
    else
        # No NVIDIA GPU — fall back to CPU-only build
        LLAMA_INSTALL_CMD = $(PIP) install llama-cpp-python --index-url=https://pypi.org/simple --force-reinstall --no-cache-dir
    endif
endif

# ---------------------------------------------------------------------------
# Pre-baked relocatable standalone-CPython distribution (built with uv).
#
# Produces a self-contained $(DIST_DIR)/ holding a standalone CPython
# (python-build-standalone, fetched via uv) with all deps installed into it,
# plus a relative-path bin/slp launcher. The directory is relocatable: tar it,
# extract anywhere, run bin/slp. llama-cpp-python is pulled PREBUILT from our
# index (see llama-wheel), so `dist` needs no compiler/CUDA toolkit:
#     make dist            # build the standalone app (downloads prebuilt llama wheel)
#     make archive         # dist + tar -> <platform>-<arch>-slp.tar.gz
# ---------------------------------------------------------------------------
PYVER ?= 3.11
DIST_DIR := dist/slp-app
ARCHIVE := $(PLATFORM)-$(ARCH)-slp.tar.gz
# Keep uv's downloaded interpreters in a build-local dir so we can glob it
# reliably on every platform (avoids parsing `uv python dir`, which on
# Windows/Git-Bash returns a backslash path).
UV_PY_DIR := $(CURDIR)/.uv-python
# Build scratch dir on the repo filesystem — /tmp is often a small tmpfs and the
# CUDA llama-cpp wheel build/zip overflows it ("Disk quota exceeded").
BUILD_TMP := $(CURDIR)/.build-tmp
# Cap llama-cpp compile parallelism so the build doesn't saturate the CPU.
# CMAKE_BUILD_PARALLEL_LEVEL is what cmake/ninja honor; MAX_JOBS for tools that
# read it. Override with `make llama-wheel MAX_JOBS=8`.
MAX_JOBS ?= 4
LLAMA_BUILD_ENV := CMAKE_BUILD_PARALLEL_LEVEL=$(MAX_JOBS) MAX_JOBS=$(MAX_JOBS)

ifeq ($(PLATFORM),windows)
    APP_PY := $(DIST_DIR)/python/python.exe
    LAUNCHER_SRC := packaging/slp-launcher.cmd
    LAUNCHER_DST := slp.cmd
else
    APP_PY := $(DIST_DIR)/python/bin/python3
    LAUNCHER_SRC := packaging/slp-launcher.sh
    LAUNCHER_DST := slp
endif

# ---------------------------------------------------------------------------
# llama-cpp-python install for the pre-baked distribution.
#
# Prebuilt source (LLAMA_PREBUILT):
#   abetlen (default) - public wheels at abetlen.github.io, per backend
#                       (cu124 / metal / cpu). Fast, no auth — but built
#                       AVX-512, so they REQUIRE an AVX-512 CPU at runtime
#                       (they SIGILL on Meteor Lake / consumer Intel 12th-gen+).
#   gcp               - our own GGML_NATIVE=OFF wheel on the private Artifact
#                       Registry: portable to any CPU; needs gcloud auth and a
#                       prior `make llama-wheel CUDA=1 PUBLISH=1`.
# LLAMA_BUILD: auto (try prebuilt, else source) | prebuilt | source.
# ---------------------------------------------------------------------------
LLAMA_VERSION ?= 0.3.25

# Backend for this host: Apple -> metal, NVIDIA/CUDA -> cu124, else cpu.
LLAMA_HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(PLATFORM),darwin)
    LLAMA_VARIANT := metal
else ifeq ($(CUDA),1)
    LLAMA_VARIANT := cu124
else ifeq ($(LLAMA_HAS_NVIDIA),yes)
    LLAMA_VARIANT := cu124
else
    LLAMA_VARIANT := cpu
endif

# A CUDA build = the cu124 variant (CUDA=1 or an NVIDIA GPU present). Used to
# decide whether `dist` bundles the CUDA 12 runtime into the relocatable archive.
ifeq ($(LLAMA_VARIANT),cu124)
    LLAMA_IS_CUDA := yes
else
    LLAMA_IS_CUDA := no
endif

# Local-version tags for OUR (gcp) wheels, e.g. 0.3.25+cu124. Off by default
# (single untagged wheel per platform); enable with LLAMA_VARIANT_TAGS=1.
LLAMA_VARIANT_TAGS ?=
ifeq ($(LLAMA_VARIANT_TAGS),1)
    LLAMA_LOCAL := +$(LLAMA_VARIANT)
    LLAMA_TAG   := $(LLAMA_VARIANT)
else
    LLAMA_LOCAL :=
    LLAMA_TAG   :=
endif

# Our private Artifact Registry python repo (publish target + the 'gcp' source).
GCP_LLAMA_INDEX := https://us-central1-python.pkg.dev/smartloop-gcp-us-east/llama-cpp/simple/
LLAMA_REPO      := $(GCP_LLAMA_INDEX:simple/=)

LLAMA_PREBUILT ?= abetlen
ifeq ($(LLAMA_PREBUILT),gcp)
    LLAMA_INDEX_HOST := $(patsubst https://%,%,$(GCP_LLAMA_INDEX))
    LLAMA_INDEX_URL   = https://oauth2accesstoken:$$(gcloud auth print-access-token)@$(LLAMA_INDEX_HOST)
    LLAMA_SPEC        = llama-cpp-python==$(LLAMA_VERSION)$(LLAMA_LOCAL)
else
    # abetlen public per-backend index (cu124 / metal / cpu) — no auth.
    LLAMA_INDEX_URL := https://abetlen.github.io/llama-cpp-python/whl/$(LLAMA_VARIANT)
    LLAMA_SPEC       = llama-cpp-python==$(LLAMA_VERSION)
endif

# Install strategy: auto = prebuilt wheel, falling back to a source build only
# if no wheel exists for this platform. NOTE: abetlen's cu124 wheels are built
# WITH AVX-512 (verified: EVEX ops in ggml_cpu_init) and SIGILL on CPUs without
# it — so on non-AVX-512 hosts use LLAMA_PREBUILT=gcp (our portable wheel) or
# LLAMA_BUILD=source. Force the path with LLAMA_BUILD=prebuilt|source.
LLAMA_BUILD ?= auto
LLAMA_PREBUILT_CMD = uv pip install --python "$(APP_PY)" --reinstall-package llama-cpp-python \
                     --only-binary llama-cpp-python --index-url="$(LLAMA_INDEX_URL)" \
                     --extra-index-url=https://pypi.org/simple --index-strategy unsafe-best-match \
                     --no-cache-dir "$(LLAMA_SPEC)"
LLAMA_SOURCE_CMD = $(LLAMA_BUILD_ENV) $(LLAMA_WHEEL_ENV) CMAKE_ARGS="$(LLAMA_WHEEL_CMAKE)" uv pip install \
                   --python "$(APP_PY)" --no-binary llama-cpp-python --reinstall-package llama-cpp-python \
                   --index-url=https://pypi.org/simple --no-cache-dir "llama-cpp-python==$(LLAMA_VERSION)"
ifeq ($(LLAMA_BUILD),source)
    LLAMA_DIST_CMD = $(LLAMA_SOURCE_CMD)
else ifeq ($(LLAMA_BUILD),prebuilt)
    LLAMA_DIST_CMD = $(LLAMA_PREBUILT_CMD)
else ifeq ($(LLAMA_BUILD),skip)
    LLAMA_DIST_CMD = :
else
    LLAMA_DIST_CMD = $(LLAMA_PREBUILT_CMD) || { echo ">> prebuilt llama wheel unavailable for $(PLATFORM)/$(ARCH) — building from source"; $(LLAMA_SOURCE_CMD); }
endif

# --- Building OUR portable wheel (make llama-wheel; publish to the gcp repo) --
WHEEL_DIR := dist-wheels
# Portable x86 baseline (no-op on arm64 / Apple Silicon, which uses NEON).
ifeq ($(ARCH),amd64)
    LLAMA_CPU_ARGS := -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF
endif
# CUDA 12.x nvcc supports host GCC only up to ~13; pick a compatible compiler.
CUDA_HOST_CXX := $(shell for v in 13 12 11; do command -v g++-$$v >/dev/null 2>&1 && { command -v g++-$$v; break; }; done)
ifneq ($(CUDA_HOST_CXX),)
    CUDA_HOST_ARGS := -DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_CXX)
    CUDA_HOST_ENV := CC=$(subst g++,gcc,$(CUDA_HOST_CXX)) CXX=$(CUDA_HOST_CXX)
endif
# Broad CUDA arch list so the wheel is portable across GPUs (Pascal..Hopper),
# with PTX on the top arch for forward-compat. Override CUDA_ARCH=89 for speed.
CUDA_ARCH ?= 61-real;70-real;75-real;80-real;86-real;89-real;90-real;90-virtual
CUDA_ARCH_ARGS := -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)

ifeq ($(PLATFORM),darwin)
    LLAMA_WHEEL_ENV   :=
    LLAMA_WHEEL_CMAKE := -DGGML_METAL=ON $(LLAMA_CPU_ARGS)
else ifeq ($(CUDA),1)
    LLAMA_WHEEL_ENV   := $(CUDA_HOST_ENV)
    LLAMA_WHEEL_CMAKE := -DGGML_CUDA=ON $(CUDA_HOST_ARGS) $(CUDA_ARCH_ARGS) $(LLAMA_CPU_ARGS)
else ifeq ($(LLAMA_HAS_NVIDIA),yes)
    LLAMA_WHEEL_ENV   := $(CUDA_HOST_ENV)
    LLAMA_WHEEL_CMAKE := -DGGML_CUDA=ON $(CUDA_HOST_ARGS) $(CUDA_ARCH_ARGS) $(LLAMA_CPU_ARGS)
else
    LLAMA_WHEEL_ENV   :=
    LLAMA_WHEEL_CMAKE := $(LLAMA_CPU_ARGS)
endif

ensure-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}

dist: ensure-uv
	@export PATH="$$HOME/.local/bin:$$PATH"; export UV_PYTHON_INSTALL_DIR="$(UV_PY_DIR)"; export UV_PYTHON_PREFERENCE=only-managed; \
	mkdir -p "$(BUILD_TMP)"; export TMPDIR="$(BUILD_TMP)"; export $(LLAMA_BUILD_ENV); set -e; \
	echo "=== [1/5] Fetching standalone CPython $(PYVER) ($(LLAMA_VARIANT)) ==="; \
	rm -rf $(DIST_DIR); mkdir -p $(DIST_DIR); \
	uv python install $(PYVER); \
	PYROOT="$$(ls -d "$(UV_PY_DIR)"/cpython-$(PYVER)*/ | sort | tail -1)"; \
	echo "Bundling standalone interpreter: $$PYROOT"; \
	cp -R "$$PYROOT." "$(DIST_DIR)/python"; \
	find "$(DIST_DIR)/python" -name EXTERNALLY-MANAGED -delete 2>/dev/null || true; \
	echo "=== [2/5] Installing requirements ==="; \
	uv pip install --python "$(APP_PY)" -r requirements.txt; \
	echo "=== [3/5] Installing llama-cpp-python ==="; \
	$(LLAMA_DIST_CMD); \
	echo "--- Bundling CUDA runtime libraries (GPU build) ---"; \
	if [ "$(LLAMA_IS_CUDA)" = "yes" ] && [ "$(PLATFORM)" = "linux" ] && [ "$(LLAMA_BUILD)" != "skip" ]; then \
	  LLAMA_LIB="$$("$(APP_PY)" -c 'import os,importlib.util as u; s=u.find_spec("llama_cpp"); print(os.path.join(os.path.dirname(s.origin),"lib"))')"; \
	  CUDA_SO="$$LLAMA_LIB/libggml-cuda.so"; \
	  [ -f "$$CUDA_SO" ] || { echo "ERROR: $$CUDA_SO missing — llama-cpp was built CPU-only (no GPU bindings)"; exit 1; }; \
	  for soname in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do \
	    src="$$(ldd "$$CUDA_SO" | awk -v s="$$soname" '$$1==s{print $$3}')"; \
	    [ -n "$$src" ] && [ -e "$$src" ] || { echo "ERROR: $$soname not resolvable at build time — install the CUDA 12 runtime"; exit 1; }; \
	    cp -aL "$$src" "$$LLAMA_LIB/$$soname"; \
	    echo "  bundled $$soname  <-  $$src"; \
	  done; \
	elif [ "$(LLAMA_IS_CUDA)" = "yes" ] && [ "$(PLATFORM)" = "windows" ] && [ "$(LLAMA_BUILD)" != "skip" ]; then \
	  LLAMA_LIB="$$(cygpath -u "$$("$(APP_PY)" -c 'import os,importlib.util as u; s=u.find_spec("llama_cpp"); print(os.path.join(os.path.dirname(s.origin),"lib"))')")"; \
	  [ -f "$$LLAMA_LIB/ggml-cuda.dll" ] || { echo "ERROR: $$LLAMA_LIB/ggml-cuda.dll missing — llama-cpp was built CPU-only (no GPU bindings)"; exit 1; }; \
	  CUDA_BIN="$$(cygpath -u "$${CUDA_PATH:-$$CUDA_HOME}")/bin"; \
	  [ -d "$$CUDA_BIN" ] || { echo "ERROR: CUDA bin dir not found (CUDA_PATH=$$CUDA_PATH CUDA_HOME=$$CUDA_HOME)"; exit 1; }; \
	  for pat in cudart64 cublas64 cublasLt64; do \
	    src="$$(ls "$$CUDA_BIN"/$${pat}_*.dll 2>/dev/null | head -1)"; \
	    [ -n "$$src" ] || { echo "ERROR: $${pat}_*.dll not found in $$CUDA_BIN — install the CUDA toolkit"; exit 1; }; \
	    cp "$$src" "$$LLAMA_LIB/"; \
	    echo "  bundled $$(basename "$$src")  <-  $$src"; \
	  done; \
	else \
	  echo "  skip (non-CUDA build)"; \
	fi; \
	echo "=== [4/5] Installing slp (non-editable) ==="; \
	uv pip install --python "$(APP_PY)" --no-deps . ; \
	echo "=== [5/5] Writing relocatable launcher ==="; \
	mkdir -p $(DIST_DIR)/bin; \
	cp $(LAUNCHER_SRC) $(DIST_DIR)/bin/$(LAUNCHER_DST); \
	chmod +x $(DIST_DIR)/bin/$(LAUNCHER_DST) 2>/dev/null || true; \
	echo "=== Smoke test ==="; \
	"$(APP_PY)" -m commands.cli --version

archive: dist
	@echo "=== Creating $(ARCHIVE) ==="
	tar -czf $(ARCHIVE) -C dist slp-app
	@echo "Archive created: $(ARCHIVE)"

# Build the portable llama-cpp-python wheel for this platform's backend
# (cu124 / metal / cpu), tag it +<variant>, and optionally publish.
#   make llama-wheel CUDA=1               # cu124 wheel (broad arch)
#   make llama-wheel CUDA=1 PUBLISH=1     # build + upload
#   make llama-wheel                      # cpu (or metal on macOS)
llama-wheel: ensure-uv
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	mkdir -p "$(BUILD_TMP)"; export TMPDIR="$(BUILD_TMP)"; set -e; \
	echo "=== Building llama-cpp-python $(LLAMA_VERSION)+$(LLAMA_VARIANT) wheel ($(PLATFORM)/$(ARCH)) ==="; \
	rm -rf .wheel-build-env $(WHEEL_DIR); \
	uv venv --seed .wheel-build-env >/dev/null; \
	$(LLAMA_BUILD_ENV) $(LLAMA_WHEEL_ENV) CMAKE_ARGS="$(LLAMA_WHEEL_CMAKE)" \
	  .wheel-build-env/bin/python -m pip wheel "llama-cpp-python==$(LLAMA_VERSION)" \
	    --no-binary llama-cpp-python --no-deps --no-cache-dir -w $(WHEEL_DIR); \
	.wheel-build-env/bin/python packaging/retag_wheel.py \
	  "$$(ls $(WHEEL_DIR)/llama_cpp_python-$(LLAMA_VERSION)-*.whl)" "$(LLAMA_TAG)" "$(WHEEL_DIR)"; \
	rm -rf .wheel-build-env; \
	echo "Built: $$(ls $(WHEEL_DIR)/*.whl)"; \
	if [ "$(PUBLISH)" = "1" ]; then $(MAKE) publish-wheel; else echo "Publish with: make publish-wheel   (or: make llama-wheel ... PUBLISH=1)"; fi

# Upload built wheels to the GCP Artifact Registry python repo. Auth via a
# short-lived gcloud access token (no keyring plugin needed); requires an active
# `gcloud auth login` / `gcloud auth application-default login`.
publish-wheel:
	@export PATH="$$HOME/.local/bin:$$PATH"; set -e; \
	test -n "$$(ls $(WHEEL_DIR)/*.whl 2>/dev/null)" || { echo "No wheels in $(WHEEL_DIR)/ — run 'make llama-wheel' first"; exit 1; }; \
	echo "=== Publishing $(WHEEL_DIR)/*.whl -> $(LLAMA_REPO) ==="; \
	uvx twine upload --repository-url $(LLAMA_REPO) \
	  --username oauth2accesstoken --password "$$(gcloud auth print-access-token)" \
	  $(WHEEL_DIR)/*.whl
	@echo "NOTE: Artifact Registry rejects re-uploading an existing version (no"
	@echo "      --skip-existing support). Bump LLAMA_VERSION or delete the old"
	@echo "      version first if you need to republish."

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python -m venv $(VENV); \
	fi

build: venv
	@echo "=== Installing requirements ==="
	$(PIP) install -r requirements.txt --index-url=https://pypi.org/simple
	@echo "=== Installing llama-cpp-python with GPU support ==="
	$(LLAMA_INSTALL_CMD)
ifeq ($(PLATFORM),windows)
	@echo "=== Build complete ==="
else
	@ln -sf .venv/bin/slp slp
	@echo "=== Build complete ==="
endif

test:
	@echo "=== Running tests ==="
	@echo "Checking slp --help..."
	@$(PYTHON) main.py --help
	@echo ""
	@echo "Checking llama-cpp-python GPU support..."
	@$(PYTHON) -c "from llama_cpp import llama_cpp; gpu=llama_cpp.llama_supports_gpu_offload(); print('GPU offload:', 'yes' if gpu else 'no (CPU-only)')"
	@echo ""
	@echo "=== Tests passed ==="

clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf *.egg-info
	@rm -rf build dist *-slp.tar.gz slp-notarize.zip $(WHEEL_DIR) .wheel-build-env $(BUILD_TMP)
	@rm -rf $(VENV)
	@echo "Cleaned"

# Code sign every Mach-O in the bundled venv (shared libs + the CPython binary).
# The launcher (bin/slp) is a shell script and is not signed.
sign:
ifeq ($(PLATFORM),darwin)
	@echo "=== Signing venv distribution ==="
	@find $(DIST_DIR) \( -name "*.so" -o -name "*.dylib" \) -print0 | xargs -0 -I {} \
		codesign --force --options runtime --timestamp \
			--entitlements packaging/macos/entitlements.mac.plist \
			--sign "$(APPLE_DEVELOPER_ID)" {}
	@codesign --force --options runtime --timestamp \
		--entitlements packaging/macos/entitlements.mac.plist \
		--sign "$(APPLE_DEVELOPER_ID)" \
		$(DIST_DIR)/python/bin/python$(PYVER)
	@echo "Signing complete"
	@codesign -v --deep $(DIST_DIR)/python/bin/python$(PYVER)
else
	@echo "Skipping code signing (not macOS)"
endif

# Notarize the zipped distribution. Apple registers the binary hashes; Gatekeeper
# verifies online on first run (a loose directory can't have a ticket stapled).
notarize:
ifeq ($(PLATFORM),darwin)
	@echo "=== Notarizing venv distribution ==="
	ditto -c -k --keepParent $(DIST_DIR) slp-notarize.zip
	xcrun notarytool submit slp-notarize.zip \
		--apple-id "$(APPLE_ID)" \
		--password "$(APPLE_APP_PASSWORD)" \
		--team-id "$(APPLE_TEAM_ID)" \
		--wait
	@rm -f slp-notarize.zip
	@echo "Notarization complete"
else
	@echo "Skipping notarization (not macOS)"
endif

release: dist sign notarize
	@echo "=== Creating $(ARCHIVE) ==="
	tar -czf $(ARCHIVE) -C dist slp-app
	@echo "=== Release v$(VERSION) complete: $(ARCHIVE) ==="

.DEFAULT_GOAL := help
