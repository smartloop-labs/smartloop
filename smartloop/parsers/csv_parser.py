"""A class for processing .csv files"""
from .base_parser import BaseParser

import os
import csv

class CsvParser(BaseParser):

    def parse(self):
        filepath = os.path.expanduser(self.file_path)
       
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        md = self.convert_to_markdown()

        return md

    def is_valid_csv(self):
        """
        Validates the CSV file structure.
        Ensures the file is not empty and all rows have the same number of columns as the header.
        """
        filepath = os.path.expanduser(self.file_path)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, mode='r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)

            if not rows:
                raise ValueError("CSV file is empty.")

            header_length = len(rows[0])
            for i, row in enumerate(rows):
                if len(row) != header_length:
                    raise ValueError(f"Row {i + 1} has an inconsistent number of columns.")

        return True

    def convert_to_markdown(self):
        """
        Converts the CSV file content to a Markdown table format.
        """
        filepath = os.path.expanduser(self.file_path)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        markdown_lines = []
        with open(filepath, mode='r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)

            if not rows:
                raise ValueError("CSV file is empty.")

            # Add header row
            header = rows[0]
            markdown_lines.append(f"| {' | '.join(header)} |")
            markdown_lines.append(f"| {' | '.join(['---'] * len(header))} |")

            # Add data rows
            for row in rows[1:]:
                markdown_lines.append(f"| {' | '.join(row)} |")

        return "\n".join(markdown_lines)

