from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"

class IngestRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    collection_name: str = Field(..., description="Collection name to store documents")
    chunk_size: int = Field(default=1000, ge=100, le=4000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Overlap between chunks")
    batch_size: int = Field(default=100, ge=10, le=1000, description="Batch size for processing")
    force_reindex: bool = Field(default=False, description="Force reindexing even if collection exists")

class IngestResponse(BaseModel):
    success: bool
    collection_name: str
    documents_processed: int
    chunks_created: int
    processing_time_seconds: float
    errors: List[str] = []

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question to answer")
    collection_name: str = Field(..., description="Collection to search in")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to retrieve")
    rerank: bool = Field(default=True, description="Whether to rerank results")
    include_metadata: bool = Field(default=True, description="Include metadata in response")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time_seconds: float
    tokens_used: Optional[int] = None

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    chunk_count: int
    created_at: datetime
    last_updated: datetime
    size_mb: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime
