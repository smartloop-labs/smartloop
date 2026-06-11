"""Creates a parser object based on the requested file's mimetype"""
import mimetypes
import logging

from .csv_parser import CsvParser

from .docx2_parser import Docx2Parser
from .pdf_parser import PdfParser

from .text_parser import TextParser
from .pptx_parser import PptxParser

from smartloop.llms import BaseLLM

class ParserFactory:
    llm: BaseLLM
    """Create an instance of a file type specific parser object based on mimetype"""
    SUPPORTED_DOCUMENT_TYPE = dict(
        pdf=mimetypes.types_map['.pdf'],
        txt=mimetypes.types_map['.txt'],
        docx="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        pptx="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        csv=mimetypes.types_map['.csv'],
        # xlsx="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        # doc=mimetypes.types_map['.doc'],  # TODO: review the support for this one
    )

    def __init__(self, llm: BaseLLM = None):
        self.llm = llm
        self.parsers = {
            self.SUPPORTED_DOCUMENT_TYPE['pdf']: PdfParser,
            self.SUPPORTED_DOCUMENT_TYPE['txt']: TextParser,
            self.SUPPORTED_DOCUMENT_TYPE['docx']: Docx2Parser,
            self.SUPPORTED_DOCUMENT_TYPE['pptx']: PptxParser,
            self.SUPPORTED_DOCUMENT_TYPE['csv']:  CsvParser,
        }

    def get_parser(self, mime_type: str, document_path: str, logger: logging.Logger):
        """Return a mime-type appropriate instance of a parser"""
        parser_class = self.parsers.get(mime_type)

        if not parser_class:
            raise ValueError(f"No parser available for mime_type: '{mime_type}'")

        parser = parser_class(self.llm, document_path, logger)
        logger.info("Created a %s for file type %s", type(parser).__name__, mime_type)

        return parser
