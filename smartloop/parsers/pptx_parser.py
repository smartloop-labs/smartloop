import re

from pptx import Presentation

from .base_parser import BaseParser

class PptxParser(BaseParser):

    def parse(self) -> str:
        full_text = self.load_text()
        paragraphs = self.split_by_paragraph(full_text)

        return "\n\n".join(paragraphs)

    # Note that this does NOT load tables right now just looks for text in text boxes
    # and other text frame type shapes
    def load_text(self) -> str:
        text = ""
        prs = Presentation(self.file_path)

        for slide in prs.slides:
            for shape in slide.shapes:
                # Check for basic text-containing shapes
                if hasattr(shape, "text"):
                    text += shape.text

                # Check for shapes with text frames
                if shape.has_text_frame:
                    text_frame = shape.text_frame
                    for paragraph in text_frame.paragraphs:
                        for run in paragraph.runs:
                            text += run.text

                # Check for other types of text-containing shapes
                if shape.shape_type in [1, 14, 17]:  # Placeholder, AutoShape, or Text Box
                    if shape.has_text_frame:
                        text_frame = shape.text_frame
                        text += text_frame.text

        return text


    def split_by_paragraph(self, text:str) -> list[str]:

        # TODO: potentially also split on "newline + indent"
        # right now we're appending \n in the text loader to ensure we at
        # least split on whole pages.
        #
        # This can be improved a lot by looking at parsing also based on newline + indent
        # or even using NLP.. at that point though we're going to want to
        # start using the LLM to parse/prepare documents too

        paragraphs = text.split('\n\n')

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
