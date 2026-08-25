import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warning
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformers warnings

# macOS App Sandbox (Mac App Store) hardening. Python's stdlib `mimetypes`
# probes a fixed list of system files at init (mimetypes.knownfiles, e.g.
# /etc/apache2/mime.types). Under the App Sandbox those paths exist but are
# read-denied, and `mimetypes` does NOT catch the resulting PermissionError:
# os.path.isfile() passes, then open() raises and the exception propagates.
# The first init is triggered transitively by importing openpyxl (via the
# document parsers), which aborts the whole `slp` process with exit code 1.
# Clear the system-file list so init relies solely on the built-in types_map,
# which is all SLP needs. Must run before any submodule import below pulls in
# openpyxl. Harmless off-sandbox / on other platforms.
import mimetypes
mimetypes.knownfiles = []

__version__ = "1.2.0"

from .conversation_store import ConversationStore, MessageInput
from .server import start_server, stop_server, get_status, is_server_running, get_server_url, read_pid_file
