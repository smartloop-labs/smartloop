import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warning
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformers warnings

__version__ = "1.0.6"

from .conversation_store import ConversationStore, MessageInput
from .server import start_server, stop_server, get_status, is_server_running, get_server_url, read_pid_file
