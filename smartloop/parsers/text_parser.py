import re

from .base_parser import BaseParser
from .document_chunk import DocumentChunk

from langchain_community.document_loaders import TextLoader

class TextParser(BaseParser):

    def parse(self) -> str:
        return self.load_text()

    def load_text(self) -> str:
        loader = TextLoader(self.file_path)
        docs = loader.load()

        return docs[0].page_content

    # TODO: this is basically a duplicate of the logic in pdf parser I think?
    # would this be useful to be hoisted up to be a common method? is it common enough?
    def split_by_paragraph(self, text:str) -> list[str]:

        # TODO: can probably improve this text splitting
        paragraphs = re.split(r'[\n][\n]+', text)

        # Process each paragraph
        processed_paragraphs = []
        for para in paragraphs:
            # Normalize whitespace: replace multiple spaces with a single space
            para = re.sub(r'\s+', ' ', para)

            # Replace remaining newlines with spaces
            para = para.replace('\n', ' ')
            para = para.strip()

            if para:
                processed_paragraphs.append(para)

        return processed_paragraphs