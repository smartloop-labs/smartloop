<p align="center">
  <a href="https://smartloop.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="img/logo.svg" />
      <source media="(prefers-color-scheme: light)" srcset="img/logo-light.svg" />
      <img width="149" height="28" alt="Smartloop" src="img/logo.svg" />
    </picture>
  </a>
</p>
<p align="center">Local agent harness for your documents. Extract information, generate content on your device</p>
<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/smartloop-ai/smartloop/build.yml"/>
</p>

<img alt="Smartloop Demo" src="demo.gif" />

### Installation

In your terminal, type in the following command to install the agent harness:

**macOS / Linux:**

```bash
curl -fsSL https://smartloop.ai/install | bash
```

**Windows (PowerShell):**

```powershell
irm https://smartloop.ai/install.ps1 | iex
```

**From source:**

> [!NOTE]
> Requires a [Rust toolchain](https://rustup.rs/) (1.85+), CMake and a C/C++ compiler with libclang (Xcode command line tools on macOS, `build-essential cmake libclang-dev` on Debian/Ubuntu, Visual Studio Build Tools + LLVM on Windows). llama.cpp is compiled into the binary with the GPU backend for your platform — Metal on macOS, Vulkan on Linux/Windows (works with NVIDIA, AMD, and Intel GPUs), which additionally needs the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

```bash
git clone https://github.com/smartloop-ai/smartloop.git
cd smartloop
make build   # cargo build --release with the platform's GPU backend (metal / vulkan)
make test    # verifies the CLI and runs the test suite
```

The binary lands at `target/release/slp`. Use `make build CARGO_FEATURES=` for a CPU-only build (no Vulkan SDK needed), or call cargo directly:

```bash
cargo build --release --features metal    # macOS
cargo build --release --features vulkan   # Linux / Windows
cargo install --path . --features metal   # install slp into ~/.cargo/bin
```

**Development:**

```bash
make run          # start the agent from source (cargo run --release -- agent start)
make run-debug    # same, with LOG_LEVEL=DEBUG
make run-timing   # same as run, timing each chat turn (see below)
cargo test        # full test suite (CPU build; model/network tests are #[ignore]d)
make test-unit    # unit tests only (lib + tests/unit)
make test-api     # HTTP API tests only (tests/api)
make build-binary # release build staged in dist/slp, as shipped by the installers
```

**Request timing:**

Set `SLP_TIMING` to see where a chat turn's time goes. `1`, `true` or `on` writes JSON lines to `~/.smartloop/logs/slp-timing.jsonl`; any other value is taken as the file path. Unset, nothing is timed or written. The supervisor, model host and project agents all append to the same file, and each turn also ends with a `[timing]` line in its `agent.log`.

```bash
make run-timing                  # SLP_TIMING=1 make run
make timing-summary              # one row per turn: phases, first token, LLM calls, model loads
make timing-summary REQ=req-1a2b3c4d   # every line of one turn, from every process
```

The packaged Studio picks it up too: `launchctl setenv SLP_TIMING 1`, then relaunch the app (`launchctl unsetenv SLP_TIMING` to stop).

Each turn has a request id that follows it from the supervisor to the project agent and the model host. A client can send its own as `X-Request-Id` to line its logs up with these; otherwise the supervisor makes one up.

In the summary, a `*` on the index column means the turn stopped waiting for indexing at the 20s cap and answered while the pass carried on; `idx overlap` is how long indexing ran alongside the answer before pausing. Below the table come models loaded more than once (reload churn between turns) and how long indexing paused for turns. Clearing logs from Studio's storage screen empties the file too.

### Using as a Rust crate

The `smartloop` library (everything behind `slp`) can be embedded in another Rust project. It is not on crates.io; each release is published to the Cargo registry at `dl.smartloop.ai/crates/`. Register it in your project's `.cargo/config.toml`:

```toml
[registries.smartloop]
index = "sparse+https://dl.smartloop.ai/crates/"
```

then depend on it from that registry:

```toml
[dependencies]
smartloop = { version = "1.2.8", registry = "smartloop" }
# Pick a GPU backend for llama.cpp:
# smartloop = { version = "1.2.8", registry = "smartloop", features = ["metal"] }
```

Available features are `metal`, `vulkan` and `cuda` (forwarded to `llama-cpp-2`); with none, llama.cpp runs on the CPU. For example, to check whether the local agent is up:

```rust
use smartloop::config::AppSettings;
use smartloop::server::{get_server_url, is_server_running};

fn main() {
    let settings = AppSettings::new();
    let (host, port) = (settings.api_host.as_str(), settings.api_port);
    println!(
        "slp {} at {} running: {}",
        smartloop::VERSION,
        get_server_url(host, port),
        is_server_running(host, port)
    );
}
```

Building the crate needs the same toolchain as building from source (CMake, libclang, and the Vulkan SDK when the `vulkan` feature is on).

### Uninstall

Use the following command to uninstall the agent harness:

**macOS / Linux:**

```bash
curl -fsSL https://smartloop.ai/uninstall | bash
```

**Windows (PowerShell):**

```powershell
irm https://smartloop.ai/uninstall.ps1 | iex
```

### Usage

To verify the installation, please type `slp --help`:

```bash
usage: slp [-h] [--version] [--resume CONVERSATION_ID] [-C PATH] {run,status,agent,token} ...

AI Agent harness for extracting information and creating new content

positional arguments:
  {run,status,agent,token}
                        Available commands
    run                 Start interactive chat session
    status              Show project status
    agent               Agent management commands (start/stop/status/restart)
    token               Manage your developer token

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --resume CONVERSATION_ID
                        Resume a previous conversation by session ID
  -C PATH, --path PATH  Working directory to use (defaults to current directory)
```


> [!TIP]
> You can generate a developer token from the [Console](https://app.smartloop.ai/).


Use it to extract information from your personal documents and do tasks for you, like writing guidelines, creating recipes, etc. Build up skills for more specialized tasks like creating events, tickets, connecting external systems, or summarizing a blog post.

You projects live inside `~/smartloop/<slug>` directory and this is where skills , documents are copied.

> [!TIP]
> You can use this tool to build up your project and consume it via the OpenAI-compatible API to build solutions on top of it.

### Requirements

| Requirement | Description |
|-------------|-------------|
| OS | macOS (Apple Silicon), Linux (x86_64, arm64), or Windows (x86_64) |
| Vulkan | GPU driver with Vulkan support (optional — GPU acceleration on Linux/Windows for NVIDIA/AMD/Intel, defaults to CPU) |

> [!NOTE]
> macOS uses Metal for GPU acceleration automatically (Vulkan via MoltenVK is available as an alternative when building with `CARGO_FEATURES="metal vulkan"`). On Windows and Linux, the pre-built release uses Vulkan and ships the Vulkan loader, so any current GPU driver (NVIDIA, AMD, or Intel) is sufficient. To build from source with Vulkan support, install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

### License

Smartloop is source-available under the [Business Source License 1.1](LICENSE) (`BUSL-1.1`).

- You may read, modify, self-host and use it in production, including inside your company.
- You may not offer it to third parties as a hosted or embedded commercial product that competes with Smartloop's paid offerings.
- Each version converts to the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) four years after it is published.

For a commercial license or questions, contact [hello@smartloop.ai](mailto:hello@smartloop.ai).
