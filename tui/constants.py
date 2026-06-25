"""tui/constants.py — Slash command registry (single source of truth)."""

from smartloop.config import canonical_model_name

VISIBLE_NAMES = {
    "slp-orchestra-mini": "SLP orchestra mini",
    "gemma3-4b": "Gemma 3 4B",
    "gemma4-2b": "Gemma 4 2B",
    "qwen2-vl-2b": "Qwen2 VL 2B",
}


def visible_model_name(key: str) -> str:
    """Return the human-readable display name for a model key."""
    return VISIBLE_NAMES.get(canonical_model_name(key), key)


LOADING_MESSAGES = [
    "[~] Warming up the engines...",
    "[+] Crunching tokens...",
    "[?] Consulting the oracle...",
    "[^] Summoning intelligence...",
    "[~] Herding cats...",
    "[>] Connecting the dots...",
    "[#] Running the numbers...",
    "[*] Thinking really hard...",
    "[!] Boiling the ocean...",
]

SLASH_COMMANDS = [
    ("/attach <path>", "Attach a file to the next message"),
    ("/document add <path>", "Add a document to the project"),
    ("/document list", "List project documents"),
    ("/document remove <#>", "Remove a document by index"),
    ("/mcp add <url>", "Register a remote MCP server"),
    ("/mcp add <command> [args]", "Register a local MCP server (e.g. uvx <pkg>)"),
    ("/mcp list", "List registered MCP servers"),
    ("/mcp remove <#>", "Remove an MCP server by index"),
    ("/project add <name>", "Create a new project"),
    ("/project list", "List all projects"),
    ("/project switch <#>", "Switch to a project by index"),
    ("/project remove <#>", "Remove a project by index"),
    ("/skill add <text>", "Add a project skill"),
    ("/skill list", "Show current skills"),
    ("/skill remove <#>", "Remove a skill by index"),
    ("/token", "Show current token status"),
    ("/token set <token>", "Set developer token"),
    ("/token clear", "Clear developer token"),
    ("/model list", "List available models"),
    ("/model enable <#>", "Enable a model by index (downloads if needed)"),
    ("/model disable <#>", "Disable a model by index"),
    ("/help", "Show this help"),
]
