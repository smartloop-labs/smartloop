.PHONY: help install build run run-debug run-single run-debug-single run-local test test-unit test-api test-cov build-binary check-apple-symbols sign notarize archive archive-zip clean clean-server clean-actions

UNAME_S := $(shell uname -s)
DIST_DIR := dist/slp

# The crate version in Cargo.toml is the single source of truth (the binary
# reports it via `slp --version`).  Read the first `version =` line of the
# [package] table.
SLP_VERSION := $(shell awk -F'"' '/^\[package\]/{p=1;next} /^\[/{p=0} p && /^version *=/{print $$2; exit}' Cargo.toml)
SERVER_PLATFORM := $(shell uname -s | tr '[:upper:]' '[:lower:]')
RAW_ARCH := $(shell uname -m)

ifeq ($(RAW_ARCH),x86_64)
    SERVER_ARCH := amd64
else ifeq ($(RAW_ARCH),aarch64)
    SERVER_ARCH := arm64
else
    SERVER_ARCH := $(RAW_ARCH)
endif
ARCHIVE_NAME := $(SERVER_PLATFORM)-$(SERVER_ARCH)-slp.tar.gz

# Releases are published by CI, not from here: .github/workflows/build.yml
# uploads every platform's archive to Cloudflare R2 (smartloop-releases),
# served at https://dl.smartloop.ai. The embedding GGUF lives there too, under
# embeddings/; it is named by model and shared across releases.
# Defaults mirror the app's slp_embedding_gguf_file / _base_url settings.
EMBED_GGUF_FILE ?= $(or $(SLP_EMBEDDING_GGUF_FILE),bge-m3-Q4_K_M.gguf)
EMBED_GGUF_BASE_URL ?= $(or $(SLP_EMBEDDING_GGUF_BASE_URL),https://dl.smartloop.ai/embeddings)
WIN_ARCHIVE_NAME := windows-amd64-slp.zip

# Detect platform
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := darwin
else
    PLATFORM := windows
endif

ifeq ($(PLATFORM),windows)
    EXE := .exe
else
    EXE :=
endif

# --- cargo ---------------------------------------------------------------------
CARGO ?= cargo
# Honour CARGO_TARGET_DIR so the binary is found wherever cargo puts it.
TARGET_DIR := $(or $(CARGO_TARGET_DIR),target)
RELEASE_BIN := $(TARGET_DIR)/release/slp$(EXE)
DIST_BIN := $(DIST_DIR)/slp$(EXE)

# GPU backends for llama.cpp (cargo features forwarded to llama-cpp-2):
# Metal on macOS, Vulkan on Linux (amd64/arm64) and Windows.  Vulkan builds
# need the Vulkan SDK (headers, loader, glslc) at build time.  Override with
# e.g. `make build CARGO_FEATURES=` for a CPU-only build, or
# `CARGO_FEATURES="metal vulkan"` on macOS to add Vulkan via MoltenVK.
ifeq ($(PLATFORM),darwin)
    CARGO_FEATURES ?= metal
else
    CARGO_FEATURES ?= vulkan
endif

# llama.cpp is compiled by llama-cpp-sys-2's build script, which forwards every
# GGML_* and CMAKE_* environment variable to CMake.  ggml would otherwise
# default to -march=native; pin a portable x86-64-v3 (AVX2) baseline on
# Linux/Windows so CI-built binaries don't emit AVX-512 and SIGILL on CPUs that
# lack it.
GGML_X86_PORTABLE := GGML_NATIVE=OFF GGML_AVX2=ON
# Linux arm64 gets the same treatment: armv8.2-a with dotprod + fp16 covers
# Graviton2+, Ampere Altra, Grace / DGX Spark and the CI's Cobalt runners
# without emitting SVE/i8mm that older cores lack.  (llama-cpp-sys-2 itself
# pins GGML_CPU_ARM_ARCH=armv8-a for non-native aarch64 Linux builds, which
# wins over the env value; either way the result is portable.)
GGML_ARM_PORTABLE := GGML_NATIVE=OFF GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
ifeq ($(PLATFORM),darwin)
    GGML_ENV :=
else ifeq ($(RAW_ARCH),aarch64)
    GGML_ENV := $(GGML_ARM_PORTABLE)
else
    GGML_ENV := $(GGML_X86_PORTABLE)
endif

# CMAKE_BUILD_PARALLEL_LEVEL parallelises the llama.cpp CMake build that
# llama-cpp-sys-2 drives.
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
CARGO_ENV = $(GGML_ENV) CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS)
CARGO_FEATURE_ARGS = $(if $(strip $(CARGO_FEATURES)),--features "$(CARGO_FEATURES)")
CARGO_BUILD = $(CARGO_ENV) $(CARGO) build --release $(CARGO_FEATURE_ARGS)
CARGO_RUN = $(CARGO_ENV) $(CARGO) run --release $(CARGO_FEATURE_ARGS) --bin slp --

# --- runtime libraries bundled next to slp -------------------------------------
# llama.cpp is linked statically, so the binary is self-contained except for
# the Vulkan loader, which the GPU driver's ICD plugs into.  Ship it so end
# users install nothing Vulkan-related:
#  - Linux: libvulkan.so.1 in dist/slp/lib/, found through the binary's
#    $ORIGIN/lib RUNPATH (set in build.rs).
#  - Windows: vulkan-1.dll next to slp.exe (the application dir is searched
#    first), plus the MSVC C++ runtime DLLs, deployed app-locally.
#  - macOS: nothing for Metal.  When built with the vulkan feature, the
#    Homebrew Vulkan loader + MoltenVK + an ICD manifest rewritten to point at
#    the bundled dylib go to dist/slp/lib/; slp aims the loader at that
#    manifest at startup (utils::runtime_env).
VULKAN_LINUX_LIB_DIRS := /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib
MSVC_RUNTIME_DLLS := vcruntime140.dll vcruntime140_1.dll msvcp140.dll

define BUNDLE_LINUX
@mkdir -p $(DIST_DIR)/lib
@found=; for d in $(VULKAN_LINUX_LIB_DIRS); do \
		if [ -e "$$d/libvulkan.so.1" ]; then \
			cp -L "$$d/libvulkan.so.1" $(DIST_DIR)/lib/ && found=1 && \
			echo "Bundled $$d/libvulkan.so.1" && break; \
		fi; \
	done; \
	if [ -z "$$found" ] && [ -n "$(filter vulkan,$(CARGO_FEATURES))" ]; then \
		echo "ERROR: libvulkan.so.1 not found (install libvulkan-dev / the Vulkan SDK)"; exit 1; \
	fi
endef

define BUNDLE_WINDOWS
@if [ -n "$(filter vulkan,$(CARGO_FEATURES))" ]; then \
		for p in "$$VULKAN_SDK/runtime/x64/vulkan-1.dll" "$${WINDIR:-C:/Windows}/System32/vulkan-1.dll"; do \
			if [ -f "$$p" ]; then cp "$$p" $(DIST_DIR)/ && echo "Bundled $$p" && break; fi; \
		done; \
		[ -f $(DIST_DIR)/vulkan-1.dll ] || { echo "ERROR: vulkan-1.dll not found (set VULKAN_SDK)"; exit 1; }; \
	fi
@for dll in $(MSVC_RUNTIME_DLLS); do \
		p="$${WINDIR:-C:/Windows}/System32/$$dll"; \
		if [ -f "$$p" ]; then cp "$$p" $(DIST_DIR)/ && echo "Bundled $$p"; fi; \
	done
endef

define BUNDLE_DARWIN
@if [ -n "$(filter vulkan,$(CARGO_FEATURES))" ]; then \
		set -e; mkdir -p $(DIST_DIR)/lib; found=; \
		for prefix in /opt/homebrew /usr/local; do \
			mvk="$$prefix/lib/libMoltenVK.dylib"; vk="$$prefix/lib/libvulkan.1.dylib"; icd=; \
			for sub in etc share; do \
				[ -f "$$prefix/$$sub/vulkan/icd.d/MoltenVK_icd.json" ] && icd="$$prefix/$$sub/vulkan/icd.d/MoltenVK_icd.json" && break; \
			done; \
			if [ -f "$$mvk" ] && [ -f "$$vk" ] && [ -n "$$icd" ]; then \
				cp -L "$$mvk" "$$vk" $(DIST_DIR)/lib/; \
				cp "$$icd" $(DIST_DIR)/lib/MoltenVK_icd.json; \
				plutil -replace ICD.library_path -string ./libMoltenVK.dylib $(DIST_DIR)/lib/MoltenVK_icd.json; \
				ref=$$(otool -L $(DIST_BIN) | awk '/libvulkan/{print $$1; exit}'); \
				[ -z "$$ref" ] || install_name_tool -change "$$ref" @executable_path/lib/libvulkan.1.dylib $(DIST_BIN); \
				found=1; echo "Bundled MoltenVK from $$prefix"; break; \
			fi; \
		done; \
		[ -n "$$found" ] || { echo "MoltenVK not found: need libMoltenVK.dylib, libvulkan.1.dylib and MoltenVK_icd.json under /opt/homebrew or /usr/local (brew install vulkan-loader molten-vk)"; exit 1; }; \
	fi
endef

# Bundle the GGUF embedding model so a packaged install works offline with no
# first-run download.  Placed at <install dir>/embeddings/<file>, which the
# embeddings loader resolves next to the executable.  Pulled from our own
# public bucket rather than the HuggingFace Hub (which gates GGUF downloads
# behind a token).  The download is retried; if it still fails, a local build
# carries on without the model (the app fetches it at runtime), but a release
# build sets REQUIRE_EMBEDDINGS=1 and fails instead: Studio's installer refuses
# an archive without it, and v1.2.20's linux-arm64 archive shipped without it
# after one connection reset.
EMBED_CACHE_DIR ?= build/embeddings
REQUIRE_EMBEDDINGS ?=
define BUNDLE_EMBEDDINGS
@mkdir -p $(EMBED_CACHE_DIR) $(DIST_DIR)/embeddings
@if [ ! -f "$(EMBED_CACHE_DIR)/$(EMBED_GGUF_FILE)" ]; then \
		echo "Downloading embedding model from $(EMBED_GGUF_BASE_URL)/$(EMBED_GGUF_FILE)"; \
		curl -fsSL --retry 5 --retry-all-errors --retry-delay 5 "$(EMBED_GGUF_BASE_URL)/$(EMBED_GGUF_FILE)" -o "$(EMBED_CACHE_DIR)/$(EMBED_GGUF_FILE).part" \
			&& mv "$(EMBED_CACHE_DIR)/$(EMBED_GGUF_FILE).part" "$(EMBED_CACHE_DIR)/$(EMBED_GGUF_FILE)" \
			|| rm -f "$(EMBED_CACHE_DIR)/$(EMBED_GGUF_FILE).part"; \
	fi
@if [ -f "$(EMBED_CACHE_DIR)/$(EMBED_GGUF_FILE)" ]; then \
		cp "$(EMBED_CACHE_DIR)/$(EMBED_GGUF_FILE)" $(DIST_DIR)/embeddings/ && \
		echo "Bundled embedding model $(EMBED_GGUF_FILE)"; \
	else \
		rmdir $(DIST_DIR)/embeddings 2>/dev/null || true; \
		if [ -n "$(REQUIRE_EMBEDDINGS)" ]; then \
			echo "ERROR: could not bundle embedding GGUF and REQUIRE_EMBEDDINGS is set."; \
			exit 1; \
		fi; \
		echo "WARNING: could not bundle embedding GGUF; the app will download it at first use."; \
	fi
endef


help:
	@echo "Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "  install         - Fetch all crate dependencies (cargo fetch)"
	@echo "  build           - Release build of slp with GPU support (features: $(CARGO_FEATURES))"
	@echo "  run             - Start the agent (one process per project) with INFO logging"
	@echo "  run-debug       - Same, with LOG_LEVEL=DEBUG (verbose orchestrator/DAG/MCP logs)"
	@echo "  run-single      - Start with one shared process for every project"
	@echo "  run-debug-single - run-single with LOG_LEVEL=DEBUG"
	@echo "  run-local       - Start the agent against the local API (http://localhost:8888)"
	@echo "  test            - Check slp --help, then run the full test suite (cargo test)"
	@echo "  clean           - Remove cargo build output, dist/ and archives"
	@echo ""
	@echo "  Testing:"
	@echo "  test-unit       - Run unit tests only (lib + tests/unit)"
	@echo "  test-api        - Run API tests only (tests/api)"
	@echo "  test-cov        - Run tests with coverage report (needs cargo-llvm-cov)"
	@echo ""
	@echo "  Packaging:"
	@echo "  build-binary    - Build slp (cargo --release, --locked) and stage it in $(DIST_DIR)"
	@echo "  check-apple-symbols - Fail if dist/ references deprecated Accelerate LAPACK APIs (macOS)"
	@echo "  sign            - Code sign the binary (macOS only)"
	@echo "  notarize        - Notarize the binary with Apple (macOS only)"
	@echo "  archive         - Create slp tar.gz from dist/slp"
	@echo "  archive-zip     - Create windows-amd64-slp.zip from dist/slp"
	@echo "  clean-server    - Remove dist/ staging and archives"
	@echo "  clean-actions   - Delete all GitHub Actions workflow runs"

install:
	@echo "=== Fetching crate dependencies ==="
	$(CARGO) fetch --locked

build:
	@echo "=== Building slp v$(SLP_VERSION) (features: $(or $(CARGO_FEATURES),none)) ==="
	$(CARGO_BUILD)
	@echo "=== Build complete: $(RELEASE_BIN) ==="

run:
	$(CARGO_RUN) agent start

run-debug:
	LOG_LEVEL=DEBUG $(CARGO_RUN) agent start

# `run` and `run-debug` are supervised: one process per project behind :38540,
# sharing one model host.  Children inherit LOG_LEVEL and write to
# <project_dir>/agent.log as well as this terminal, so the orchestrator trace
# still shows up live.  These two fall back to the single shared process.
run-single:
	$(CARGO_RUN) agent start --no-supervised

run-debug-single:
	LOG_LEVEL=DEBUG $(CARGO_RUN) agent start --no-supervised

run-local:
	SLP_API_BASE=http://localhost:8888 $(CARGO_RUN) agent start

# Tests build without GPU features (CPU llama.cpp) so they run on any host.
test:
	@echo "=== Running tests ==="
	@echo "Checking slp --help..."
	@$(CARGO) run --quiet --bin slp -- --help
	@echo ""
	$(CARGO) test

test-unit:
	$(CARGO) test --lib --test unit

test-api:
	$(CARGO) test --test api

test-cov:
	@command -v cargo-llvm-cov >/dev/null 2>&1 || { echo "ERROR: install cargo-llvm-cov (cargo install cargo-llvm-cov)"; exit 1; }
	$(CARGO) llvm-cov --all-targets --summary-only

build-binary:
	@echo "Building slp v$(SLP_VERSION) with cargo (features: $(or $(CARGO_FEATURES),none))..."
	SLP_PACKAGED_BUILD=1 $(CARGO_BUILD) --locked
	@rm -rf $(DIST_DIR)
	@mkdir -p $(DIST_DIR)
	cp $(RELEASE_BIN) $(DIST_BIN)
ifeq ($(PLATFORM),linux)
	$(BUNDLE_LINUX)
else ifeq ($(PLATFORM),darwin)
	$(BUNDLE_DARWIN)
else
	$(BUNDLE_WINDOWS)
endif
	$(BUNDLE_EMBEDDINGS)
	@echo "Verifying binary..."
	@$(DIST_BIN) --version || (echo "ERROR: slp binary verification failed" && exit 1)
	@echo "Binary verified successfully."

# Guard against the App Store "non-public or deprecated APIs" rejection that
# blocks Studio's Mac App Store build, which bundles this framework wholesale.
# The offenders are the legacy Accelerate LAPACK helpers that have no
# $NEWLAPACK replacement, pulled in when a dependency links Accelerate's LAPACK
# rather than shipping its own BLAS (llama.cpp is built with GGML_BLAS=OFF).
# Scans every Mach-O in dist/ — the slp binary itself and any bundled dylib —
# for an *undefined* reference; a library that defines the symbol itself is
# not an Apple API reference and is fine.
DEPRECATED_APPLE_SYMS := _dcabs1_|_lsame_|_xerbla_array__

check-apple-symbols:
ifeq ($(PLATFORM),darwin)
	@echo "=== Scanning $(DIST_DIR) for deprecated Accelerate LAPACK symbols ==="
	@hits=$$(find $(DIST_DIR) -type f -print0 | xargs -0 file | grep 'Mach-O' | cut -d: -f1 \
		| while read -r bin; do \
			nm -u -arch all "$$bin" 2>/dev/null | grep -qE "^ *($(DEPRECATED_APPLE_SYMS))$$" && echo "$$bin"; \
		done | sort); \
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
	@# signature with hardened runtime, so sign every bundled Mach-O (MoltenVK /
	@# Vulkan loader in a Vulkan build) before the slp binary itself.
	@find $(DIST_DIR) -type f ! -path "$(DIST_BIN)" -print0 | xargs -0 file | \
		grep 'Mach-O' | cut -d: -f1 | while read -r bin; do \
		codesign --force --options runtime --timestamp \
			--entitlements packaging/macos/entitlements.mac.plist \
			--sign "$(APPLE_DEVELOPER_ID)" "$$bin" || exit 1; \
	done
	codesign --force --options runtime --timestamp \
		--entitlements packaging/macos/entitlements.mac.plist \
		--sign "$(APPLE_DEVELOPER_ID)" \
		$(DIST_BIN)
	@echo "Signing complete"
	codesign -v $(DIST_BIN)
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
	ID=$$(plutil -extract id raw -o - notarize-result.json); \
	STATUS=$$(plutil -extract status raw -o - notarize-result.json); \
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

# Same layout as the tarballs: a top-level slp/ folder holding slp.exe.
# 7-Zip ships on the GitHub Windows runners; fall back to zip, then to the
# Windows bsdtar (tar.exe -a picks the zip format from the extension).
archive-zip:
	@echo "Packaging $(WIN_ARCHIVE_NAME)..."
	@rm -f $(WIN_ARCHIVE_NAME)
	@if command -v 7z >/dev/null 2>&1; then \
		cd dist && 7z a -tzip -bso0 -bsp0 ../$(WIN_ARCHIVE_NAME) slp; \
	elif command -v zip >/dev/null 2>&1; then \
		cd dist && zip -qr ../$(WIN_ARCHIVE_NAME) slp; \
	else \
		"$${SYSTEMROOT:-C:/Windows}/System32/tar.exe" -a -cf $(WIN_ARCHIVE_NAME) -C dist slp; \
	fi
	@echo "Created $(WIN_ARCHIVE_NAME)"

clean-server:
	@echo "Cleaning packaging artifacts..."
	@rm -rf $(DIST_DIR) slp-*.tar.gz *-slp.tar.gz $(WIN_ARCHIVE_NAME) slp-notarize.zip notarize-result.json
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
	@echo "Cleaning cargo build output..."
	$(CARGO) clean
	@rm -rf build dist $(ARCHIVE_NAME)
	@echo "Cleaned"

.DEFAULT_GOAL := help
