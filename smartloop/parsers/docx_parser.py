"""A class for processing .doc and .docx files in XML format"""
import os
import logging

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

from .base_parser import BaseParser

class DocxParser(BaseParser):
    """
    A class for processing .doc and .docx files in XML format
    """

    def parse(self):
        paragraphs = self.load_paragraphs()
        return "\n\n".join(paragraphs)

    def is_list_item(self, paragraph) -> bool:
        """
        Determines if a paragraph is a list item.
        Uses indentation and numbering/bullets to infer list items.
        """
        if paragraph.style.name.startswith("List") or paragraph.alignment == WD_PARAGRAPH_ALIGNMENT.LEFT:
            return True
        return False

    def load_paragraphs(self) -> list[str]:
        doc = Document(self.file_path)
        text_blocks = []
        current_block = None
        carry_over_heading = None

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue  # Skip empty paragraphs

            # Check if the paragraph is a heading
            if paragraph.style.name.startswith("Heading"):
                if current_block and carry_over_heading:
                    # Merge carry-over heading with the current block
                    current_block = f"{carry_over_heading} {current_block.strip()}"
                    carry_over_heading = None

                # Handle deferred heading
                if current_block and len(current_block.split()) <= 2:
                    carry_over_heading = current_block.strip()
                    current_block = None

                # Start a new block for the heading
                if current_block:
                    text_blocks.append(current_block.strip())
                current_block = text
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info("Started new text block with heading: %s", text)

            # Check if the paragraph is a list item
            elif self.is_list_item(paragraph):
                if carry_over_heading:
                    # Merge the carry-over heading with the list item
                    current_block = f"{carry_over_heading} {text}"
                    carry_over_heading = None
                elif current_block:
                    current_block += f" {text}"
                else:
                    current_block = text

                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info("Appended list item to current block: %s", text)

            # Append to the current block for regular paragraphs
            else:
                if carry_over_heading:
                    # Merge the carry-over heading into the current block
                    current_block = f"{carry_over_heading} {text}"
                    carry_over_heading = None
                elif current_block:
                    current_block += f" {text}"
                else:
                    current_block = text

                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info("Appended text to current block: %s", text)

        # Finalize any remaining blocks
        if carry_over_heading and current_block:
            current_block = f"{carry_over_heading} {current_block.strip()}"
        if current_block:
            text_blocks.append(current_block.strip())

        return text_blocks
