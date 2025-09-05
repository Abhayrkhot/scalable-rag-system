import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger()

class EmbeddingService:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.dimension = settings.embedding_dimension
        
        # Initialize embedding model
        if self.embedding_model.startswith("text-embedding"):
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=settings.openai_api_key
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def embed_documents_batch(self, documents: List[Document]) -> List[List[float]]:
        """Embed a batch of documents with retry logic"""
        try:
            texts = [doc.page_content for doc in documents]
            
            if self.embedding_model.startswith("text-embedding"):
                embeddings = await self._embed_with_openai(texts)
            else:
                embeddings = await self._embed_with_sentence_transformers(texts)
            
            logger.info(f"Successfully embedded {len(documents)} documents")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    async def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI API"""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using sentence transformers (runs in thread pool)"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            self.embeddings.embed_documents, 
            texts
        )
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        try:
            if self.embedding_model.startswith("text-embedding"):
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=[query]
                )
                return response.data[0].embedding
            else:
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    self.embeddings.embed_query,
                    query
                )
                return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    async def embed_documents_async(self, documents: List[Document]) -> List[Document]:
        """Embed documents asynchronously in batches"""
        all_embeddings = []
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_embeddings = await self.embed_documents_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Embedded batch {i//self.batch_size + 1}/{(len(documents)-1)//self.batch_size + 1}")
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, all_embeddings):
            doc.metadata["embedding"] = embedding
        
        return documents
