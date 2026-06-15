import logging
import mimetypes
from typing import Optional

from smartloop.llms import BaseLLM
from smartloop.parsers import ParserFactory

logger = logging.getLogger(__name__)


class FileOperations:
    """Binary document parsing for the explorer.

    File read/write/grep/bash now run through the smartloop system-tools MCP
    server (see ``smartloop/api/mcp/system_server.py``); the only direct
    helper still needed is binary parsing, which requires an LLM and so
    cannot live in the stdio MCP subprocess.
    """

    BINARY_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv"}

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def parse_binary_file(self, file_path: str) -> Optional[str]:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            return None

        try:
            factory = ParserFactory(llm=self.llm)
            parser = factory.get_parser(mime_type, file_path, logger)
            return parser.parse()
        except ValueError:
            logger.warning(f"[file-read] No parser for mime type: {mime_type}")
            return None
        except Exception as e:
            logger.error(f"[file-read] Parser failed for {file_path}: {e}")
            return None
