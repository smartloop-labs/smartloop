"""Creates a parser object based on the requested file's mimetype"""
import mimetypes
import logging

from .csv_parser import CsvParser

from .document_parser import DocumentParser
from .docx2_parser import Docx2Parser
from .pdf_parser import PdfParser

from .json_parser import JsonParser, looks_like_json
from .text_parser import TextParser

from smartloop.llms import BaseLLM

class ParserFactory:
    llm: BaseLLM
    """Create an instance of a file type specific parser object based on mimetype"""
    SUPPORTED_DOCUMENT_TYPE = dict(
        pdf=mimetypes.types_map['.pdf'],
        txt=mimetypes.types_map['.txt'],
        docx="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        pptx="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        xlsx="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        html=mimetypes.types_map['.html'],
        csv=mimetypes.types_map['.csv'],
        json="application/json",
        png="image/png",
        jpeg="image/jpeg",
    )

    def __init__(self, llm: BaseLLM = None):
        self.llm = llm
        # PDF, DOCX, PPTX, XLSX, HTML and images are all converted to markdown
        # via MarkItDown (DocumentParser); PDF/DOCX keep thin subclasses for
        # readability.  Images (PNG/JPEG) are OCR'd/described by the vision
        # model through MarkItDown's image converter.  TXT and CSV use their own
        # lightweight parsers; JSON (and JSON-bodied TXT) is rendered to Markdown
        # by JsonParser since MarkItDown returns JSON verbatim.
        self.parsers = {
            self.SUPPORTED_DOCUMENT_TYPE['pdf']: PdfParser,
            self.SUPPORTED_DOCUMENT_TYPE['txt']: TextParser,
            self.SUPPORTED_DOCUMENT_TYPE['docx']: Docx2Parser,
            self.SUPPORTED_DOCUMENT_TYPE['pptx']: DocumentParser,
            self.SUPPORTED_DOCUMENT_TYPE['xlsx']: DocumentParser,
            self.SUPPORTED_DOCUMENT_TYPE['html']: DocumentParser,
            self.SUPPORTED_DOCUMENT_TYPE['json']: JsonParser,
            self.SUPPORTED_DOCUMENT_TYPE['png']: DocumentParser,
            self.SUPPORTED_DOCUMENT_TYPE['jpeg']: DocumentParser,
            self.SUPPORTED_DOCUMENT_TYPE['csv']:  CsvParser,
        }

    def get_parser(self, mime_type: str, document_path: str, logger: logging.Logger):
        """Return a mime-type appropriate instance of a parser"""
        parser_class = self.parsers.get(mime_type)

        # A .txt whose body is actually JSON is routed to JsonParser so it gets
        # structured into Markdown instead of stored as a raw blob.
        if (parser_class is TextParser
                and looks_like_json(document_path)):
            parser_class = JsonParser

        if not parser_class:
            raise ValueError(f"No parser available for mime_type: '{mime_type}'")

        parser = parser_class(self.llm, document_path, logger)
        logger.info("Created a %s for file type %s", type(parser).__name__, mime_type)

        return parser
