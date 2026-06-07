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
> Requires the `CUDA` toolkit. For NVIDIA GPU acceleration on Linux/Windows, install [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) before proceeding. The build step auto-creates and activates a virtual environment, then detects your GPU backend — Metal on macOS, CUDA on Linux/Windows with NVIDIA.

```bash
git clone https://github.com/smartloop-ai/smartloop.git
cd smartloop
make build   # creates and activates .venv, installs dependencies, and configures GPU backend
make test    # verifies CLI and GPU offload support
```

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
| OS | macOS (Apple Silicon), Linux (x86_64), or Windows (x86_64) |
| CUDA | 12.4+ (optional — NVIDIA GPU acceleration on Linux/Windows, defaults to CPU) |

> [!NOTE]
> macOS uses Metal for GPU acceleration automatically. On Windows and Linux, the pre-built release bundles the CUDA runtime, so only the NVIDIA driver is required. To build from source with GPU support, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### License

© 2026 Smartloop Inc.

All code is licensed under Apache 2.0. See [LICENSE](LICENSE) file for details.
