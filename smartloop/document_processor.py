"""
DocumentProcessor
"""
import asyncio
import json
import uuid
import os
import tempfile
import mimetypes
import re

from bson import ObjectId
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from smartloop.parsers.document_chunk import DocumentChunk
from smartloop.parsers.parser_factory import ParserFactory
from smartloop.llms import BaseLLM
from smartloop.utils.chunk_converter import convert_chunks_to_document_chunks

# TODO: where to put this?
class DocumentMissingError(Exception):
    '''Raise when document is missing'''

class ChunkOutput(BaseModel):
    """Pydantic model for chunk output"""
    chunks: List[Dict[str, Any]] = Field(description="List of document chunks")

class Summary(BaseModel):
    id: str
    text: str
    num_of_tokens: int

IS_TOKEN_LIMIT_REACHED = 'is_token_limit_reached'
NUM_OF_TOKENS = 'num_of_tokens'

class DocumentProcessor:
    """
    Processor that collects all pending documents and then parses and stores them vector storage
    """

    def __init__(self, llm: BaseLLM, projects, documents, gridfs, vector_db, logger, section_token_size: int = 512):
        """
        Initialize the DocumentProcessor with a MongoDB connection.
        """
        self.llm = llm
        self.projects = projects
        self.documents = documents
        self.gridfs = gridfs
        self.vector_db = vector_db
        self.logger = logger
        self.section_token_size = section_token_size


    def process_pending_documents(self, project_id: str):
        """
        Find all currently pending Documents and initiate processing on them
        """
        self.logger.info("processing pending docs...")

        pending_documents = self.documents.get_all_pending(project_id)

        self.logger.info("Found pending documents: %s", pending_documents)

        for document in pending_documents:
            asyncio.run(self.process_document(document))


    async def process_document(self, document):
        """
        Process an individual document
        """

        try:
            project_id = document.get('project_id')
            deleted = document.get('is_deleted', False)
            # process if not deleted
            if not deleted:
                project = self.projects.get(project_id)

                if project is not None:
                    dat = self.gridfs.get(document.get('file_id'))

                    # Extract project ID from document path (first part before the first slash)
                    project_id_from_path = document.get('path').split('/')[0]
                    document_dir = os.path.join(tempfile.tempdir, project_id_from_path)

                    if not os.path.isdir(document_dir):
                        os.makedirs(document_dir)

                    filepath = os.path.join(tempfile.tempdir, document.get('path'))

                    with open(filepath, 'wb+') as file:
                        file.write(dat.read())

                    self.process_document_as_chunks(project, document)

                    document['is_pending'] = False
                    self.documents.update(document)
            else:
                # If marked deleted then the project was deleted so delete all chunks for this document
                document_id = document.get('_id')
                # init the collection
                self.vector_db.open_project(str(project_id))
                self.vector_db.delete_all_chunks_by_document_id(document_id=str(document_id))

        except DocumentMissingError as doc_missing_error:
            self.logger.error(doc_missing_error, exc_info=True)
        except Exception as error:
            self.logger.error(error, exc_info=True)
        finally:
            document['is_pending'] = False
            self.documents.update(document)

    def count_processed_tokens_for_project(self, project_id: str, document_id: ObjectId) -> int:
        return np.sum(np.array([
            doc.get('num_of_tokens', 0)
            for doc in self.documents.find_by_project(project_id)
            if doc.get('_id') != document_id
        ]))


    def get_sections(self, text: str, target_tokens_per_section: int):
        """
        Split document text into sections ensuring each section doesn't exceed max_tokens_per_section
        First tries to split by markdown headers, then by paragraphs, then by sentences if needed

        Args:
            text: The document text to split
            max_tokens_per_section: Maximum tokens allowed per section (default: uses instance setting)

        Returns:
            List of text sections, each within the token limit
        """
        sections = []

        # First, try to split by markdown headers if they exist
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        has_headers = any(re.match(header_pattern, line, re.MULTILINE) for line in lines)

        if has_headers:
            # Process markdown sections
            markdown_sections = []
            current_section = []
            current_header_level = 0

            for line in lines:
                header_match = re.match(header_pattern, line, re.MULTILINE)

                if header_match:
                    header_level = len(header_match.group(1))

                    if current_section and (header_level <= current_header_level or current_header_level == 0):
                        section_content = '\n'.join(current_section).strip()
                        if section_content:
                            markdown_sections.append(section_content)
                        current_section = []

                    current_section.append(line)
                    current_header_level = header_level
                else:
                    current_section.append(line)

            if current_section:
                section_content = '\n'.join(current_section).strip()
                if section_content:
                    markdown_sections.append(section_content)

            # Check if any markdown section exceeds token limit
            for section in markdown_sections:
                sections.extend(self._split_large_section(section, target_tokens_per_section))
        else:
            # No markdown headers, split by other methods
            sections.extend(self._split_large_section(text, target_tokens_per_section))

        self.logger.debug(f"Split document into {len(sections)} sections, each under {target_tokens_per_section} tokens")
        return sections

    def _split_large_section(self, text: str, max_tokens: int) -> List[str]:
        """
        Split a large text section into smaller chunks that don't exceed max_tokens
        Uses paragraph boundaries, then sentence boundaries, then character boundaries
        """
        sections = []

        # If the entire section is within limit, return as-is
        if self.llm.get_num_tokens(text) <= max_tokens:
            return [text.strip()] if text.strip() else []

        # Try splitting by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) > 1:
            current_chunk = ""
            current_tokens = 0

            for paragraph in paragraphs:
                paragraph_tokens = self.llm.get_num_tokens(paragraph)

                # If single paragraph exceeds limit, split it further
                if paragraph_tokens > max_tokens:
                    # Save current chunk if it has content
                    if current_chunk:
                        sections.append(current_chunk.strip())
                        current_chunk = ""
                        current_tokens = 0

                    # Split the large paragraph
                    sections.extend(self._split_by_sentences(paragraph, max_tokens))
                    continue

                # If adding this paragraph would exceed limit, save current chunk
                if current_tokens + paragraph_tokens > max_tokens and current_chunk:
                    sections.append(current_chunk.strip())
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens

            # Add the last chunk
            if current_chunk:
                sections.append(current_chunk.strip())
        else:
            # Single paragraph or no paragraph breaks, split by sentences
            sections.extend(self._split_by_sentences(text, max_tokens))

        return sections

    def _split_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text by sentences, keeping within token limits
        """
        import re

        # Simple sentence splitting (can be improved with nltk if needed)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sections = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self.llm.get_num_tokens(sentence)

            # If single sentence exceeds limit, split by characters
            if sentence_tokens > max_tokens:
                # Save current chunk if it has content
                if current_chunk:
                    sections.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                # Split by characters as last resort
                sections.extend(self._split_by_characters(sentence, max_tokens))
                continue

            # If adding this sentence would exceed limit, save current chunk
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                sections.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens

        # Add the last chunk
        if current_chunk:
            sections.append(current_chunk.strip())

        return sections

    def _split_by_characters(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text by characters as a last resort, keeping within token limits
        """
        sections = []
        words = text.split()
        current_chunk = ""
        current_tokens = 0

        for word in words:
            word_tokens = self.llm.get_num_tokens(word)

            if current_tokens + word_tokens > max_tokens and current_chunk:
                sections.append(current_chunk.strip())
                current_chunk = word
                current_tokens = word_tokens
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
                current_tokens += word_tokens

        if current_chunk:
            sections.append(current_chunk.strip())

        return sections

    def process_large_document_in_sections(self, document_text: str, document_id: str) -> list[DocumentChunk]:
        """
        Process a large document by breaking it into sections and chunking each section separately
        Maximizes token usage per section by combining smaller sections together

        Args:
            document_text: The full document text
            document_id: The document ID

        Returns:
            List of all DocumentChunk objects from all sections
        """
        all_chunks = []
        max_tokens = self.llm.get_max_tokens()
        # Use configured section token size as the target tokens per section
        target_tokens_per_section = self.section_token_size if max_tokens > self.section_token_size else max_tokens

        self.logger.info(f"Processing large document in sections for document_id: {document_id}")
        self.logger.info(f"Target tokens per section: {target_tokens_per_section}")

        # Split document into sections with configured token limit
        sections = self.get_sections(document_text, target_tokens_per_section)
        self.logger.info(f"Document split into {len(sections)} sections")

        for section_idx, section in enumerate(sections):
            if not section.strip():
                continue
            section_tokens = self.llm.get_num_tokens(section)
            self.logger.info(f"Processing section {section_idx+1}/{len(sections)} with {section_tokens} tokens")
            try:
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_id=str(uuid.uuid4()),
                    paragraph_id=f"{document_id}_{section_idx}",
                    text=section,
                    metadata=json.dumps({
                        'token_count': section_tokens
                    }),
                    total_chunks=len(sections)
                )
                all_chunks.append(chunk)
                self.logger.info(f"Section {section_idx+1} converted to chunk with {section_tokens} tokens")
            except Exception as e:
                self.logger.error(f"Error processing section {section_idx+1}: {e}")
                # Do not append a string fallback, only log the error

        self.logger.info(f"Total chunks generated from all sections: {len(all_chunks)}")
        return all_chunks

    def process_document_as_chunks(self, project: dict, document: dict):
        """
        Create a doc specific parser and parse the document into DocumentChunk objects
        Then insert the chunks into the vectorDB, which creates the embeddings in the process
        """
        try:
            document_path = os.path.join(tempfile.tempdir, document.get('path'))
            mime_type = mimetypes.guess_type(document_path)
            document_id = str(document.get('_id'))
            project_id = str(project.get('_id'))

            # Parse the document into chunks
            parser = ParserFactory(self.llm).get_parser(mime_type[0], document_path, self.logger)
            document_text = parser.parse()

            num_of_tokens = self.llm.get_num_tokens(document_text)

            document_chunks = self.process_large_document_in_sections(document_text, document_id)

            self.vector_db.open_project(project_id)
            self.vector_db.upsert_document_chunks(document_chunks)

            # Build a quick-view summary from the first paragraph of the document
            first_para = next(
                (p.strip() for p in document_text.split('\n\n') if p.strip()),
                document_text.strip()
            )
            document['summary'] = first_para[:300] if first_para else None
            document[IS_TOKEN_LIMIT_REACHED] = False
            document[NUM_OF_TOKENS] = num_of_tokens
        except ValueError as ve:
            raise DocumentMissingError(ve) from ve
        except Exception as ex:
            raise ex
        finally:
            document['pending'] = False
            document['processing'] = False
            self.documents.update(document)
