import os
import asyncio
import aiofiles
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.schema import Document
import structlog

logger = structlog.get_logger()

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    async def process_files(self, file_paths: List[str], batch_size: int = 100) -> List[Document]:
        """Process multiple files asynchronously in batches"""
        all_documents = []
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batch_docs = await self._process_batch(batch)
            all_documents.extend(batch_docs)
            logger.info(f"Processed batch {i//batch_size + 1}, total docs: {len(all_documents)}")
        
        return all_documents
    
    async def _process_batch(self, file_paths: List[str]) -> List[Document]:
        """Process a batch of files concurrently"""
        tasks = [self.process_single_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {file_paths[i]}: {result}")
            else:
                documents.extend(result)
        
        return documents
    
    async def process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file and return list of documents"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit
                raise ValueError(f"File too large: {file_size_mb:.2f}MB")
            
            # Extract text based on file type
            text = await self._extract_text(file_path)
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return []
            
            # Create metadata
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "file_size_mb": round(file_size_mb, 2),
                "text_length": len(text)
            }
            
            # Split text into chunks
            documents = self._split_text(text, metadata)
            
            logger.info(f"Processed {file_path}: {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return await self._extract_pdf_text(file_path)
        elif suffix in ['.md', '.markdown']:
            return await self._extract_markdown_text(file_path)
        elif suffix in ['.txt', '.text']:
            return await self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF using PyMuPDF for better performance"""
        try:
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {file_path}, trying PyPDF2: {e}")
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
    
    async def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from markdown file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()
    
    async def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()
    
    def _split_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into chunks using appropriate splitter"""
        # Use markdown splitter for markdown files
        if metadata.get("file_type") in ['.md', '.markdown']:
            chunks = self.markdown_splitter.split_text(text)
        else:
            chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata.update({
                "chunk_index": i,
                "chunk_size": len(chunk),
                "token_count": len(self.encoding.encode(chunk))
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
