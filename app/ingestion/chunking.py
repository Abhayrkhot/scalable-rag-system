import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.schema import Document
import hashlib

logger = structlog.get_logger()

class SectionAwareChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize splitters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Section patterns
        self.heading_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^\d+\.\s+(.+)$',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headings
            r'^[A-Z][a-z\s]+:$',  # Title case with colon
        ]
        
        self.page_break_patterns = [
            r'^\s*---+\s*$',  # Horizontal rules
            r'^\s*\*\*\*\s*$',  # Asterisk breaks
            r'^\s*Page\s+\d+\s*$',  # Page markers
            r'^\s*\[Page\s+\d+\]\s*$',  # Bracket page markers
        ]
    
    def chunk_document(self, document: Document, file_path: str) -> List[Document]:
        """Chunk document with section awareness"""
        try:
            # Extract document metadata
            doc_metadata = self._extract_document_metadata(document, file_path)
            
            # Split into sections first
            sections = self._split_into_sections(document.page_content)
            
            # Chunk each section
            all_chunks = []
            for section in sections:
                section_chunks = self._chunk_section(section, doc_metadata)
                all_chunks.extend(section_chunks)
            
            # Add content hash to each chunk
            for chunk in all_chunks:
                chunk.metadata["content_hash"] = self._compute_content_hash(chunk)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {file_path}: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(document, file_path)
    
    def _extract_document_metadata(self, document: Document, file_path: str) -> Dict[str, Any]:
        """Extract document-level metadata"""
        file_path_obj = Path(file_path)
        
        metadata = {
            "source": file_path,
            "file_name": file_path_obj.name,
            "file_type": file_path_obj.suffix.lower(),
            "doc_title": self._extract_document_title(document.page_content),
            "total_length": len(document.page_content),
            "created_at": document.metadata.get("created_at"),
            "author": document.metadata.get("author"),
            "version": document.metadata.get("version")
        }
        
        return metadata
    
    def _extract_document_title(self, content: str) -> str:
        """Extract document title from content"""
        lines = content.split('\n')
        
        # Look for title in first few lines
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            
            # Check for markdown title
            if line.startswith('# '):
                return line[2:].strip()
            
            # Check for title case line
            if len(line) < 100 and line[0].isupper() and not line.endswith('.'):
                return line
        
        # Fallback to first non-empty line
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:
                return line
        
        return "Untitled Document"
    
    def _split_into_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split content into sections with metadata"""
        sections = []
        lines = content.split('\n')
        
        current_section = {
            "title": "Introduction",
            "content": "",
            "level": 0,
            "page_num": 1,
            "section_num": 0
        }
        
        page_num = 1
        section_num = 0
        
        for line in lines:
            # Check for page breaks
            if self._is_page_break(line):
                page_num += 1
                continue
            
            # Check for new section
            section_info = self._detect_section(line)
            if section_info:
                # Save current section
                if current_section["content"].strip():
                    current_section["section_num"] = section_num
                    sections.append(current_section.copy())
                    section_num += 1
                
                # Start new section
                current_section = {
                    "title": section_info["title"],
                    "content": line + "\n",
                    "level": section_info["level"],
                    "page_num": page_num,
                    "section_num": section_num
                }
            else:
                # Add line to current section
                current_section["content"] += line + "\n"
        
        # Add final section
        if current_section["content"].strip():
            current_section["section_num"] = section_num
            sections.append(current_section)
        
        return sections
    
    def _is_page_break(self, line: str) -> bool:
        """Check if line is a page break"""
        for pattern in self.page_break_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _detect_section(self, line: str) -> Optional[Dict[str, Any]]:
        """Detect if line starts a new section"""
        line = line.strip()
        if not line:
            return None
        
        for pattern in self.heading_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                level = self._get_heading_level(line)
                return {
                    "title": title,
                    "level": level
                }
        
        return None
    
    def _get_heading_level(self, line: str) -> int:
        """Get heading level (1-6 for markdown, 1 for others)"""
        if line.startswith('#'):
            return min(6, len(line) - len(line.lstrip('#')))
        return 1
    
    def _chunk_section(self, section: Dict[str, Any], doc_metadata: Dict[str, Any]) -> List[Document]:
        """Chunk a single section"""
        content = section["content"].strip()
        if not content:
            return []
        
        # Choose appropriate splitter
        if doc_metadata["file_type"] in [".md", ".markdown"]:
            chunks = self.markdown_splitter.split_text(content)
        else:
            chunks = self.text_splitter.split_text(content)
        
        # Create documents with section metadata
        documents = []
        for i, chunk_content in enumerate(chunks):
            metadata = {
                **doc_metadata,
                "section_title": section["title"],
                "section_level": section["level"],
                "section_num": section["section_num"],
                "page_num": section["page_num"],
                "chunk_index": i,
                "chunk_size": len(chunk_content),
                "is_section_start": i == 0,
                "is_section_end": i == len(chunks) - 1
            }
            
            documents.append(Document(
                page_content=chunk_content,
                metadata=metadata
            ))
        
        return documents
    
    def _compute_content_hash(self, document: Document) -> str:
        """Compute SHA256 hash of normalized content"""
        # Normalize content
        normalized = self._normalize_text(document.page_content)
        
        # Include metadata in hash for uniqueness
        metadata_str = str(sorted(document.metadata.items()))
        content = f"{normalized}|||{metadata_str}"
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common punctuation that might vary
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def _fallback_chunking(self, document: Document, file_path: str) -> List[Document]:
        """Fallback chunking when section-aware chunking fails"""
        logger.warning(f"Using fallback chunking for {file_path}")
        
        # Use simple text splitter
        chunks = self.text_splitter.split_text(document.page_content)
        
        documents = []
        file_path_obj = Path(file_path)
        
        for i, chunk_content in enumerate(chunks):
            metadata = {
                "source": file_path,
                "file_name": file_path_obj.name,
                "file_type": file_path_obj.suffix.lower(),
                "doc_title": "Unknown",
                "section_title": "Unknown",
                "section_level": 0,
                "section_num": 0,
                "page_num": 1,
                "chunk_index": i,
                "chunk_size": len(chunk_content),
                "is_section_start": False,
                "is_section_end": False
            }
            
            documents.append(Document(
                page_content=chunk_content,
                metadata=metadata
            ))
        
        return documents

class ChunkingPipeline:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunker = SectionAwareChunker(chunk_size, chunk_overlap)
    
    def process_documents(self, documents: List[Document], file_paths: List[str]) -> List[Document]:
        """Process multiple documents through chunking pipeline"""
        all_chunks = []
        
        for doc, file_path in zip(documents, file_paths):
            try:
                chunks = self.chunker.chunk_document(doc, file_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return all_chunks
