import pytest
import asyncio
from pathlib import Path
from app.services.ingestion_service import IngestionService
from app.core.document_processor import DocumentProcessor

@pytest.fixture
def sample_docs():
    """Create sample documents for testing"""
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample text file
    (sample_dir / "test1.txt").write_text("This is a test document about artificial intelligence and machine learning.")
    
    # Create sample markdown file
    (sample_dir / "test2.md").write_text("""
# Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms.

## Types of ML

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning
""")
    
    return [str(sample_dir / "test1.txt"), str(sample_dir / "test2.md")]

@pytest.mark.asyncio
async def test_document_processor(sample_docs):
    """Test document processing"""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    documents = await processor.process_files(sample_docs, batch_size=10)
    
    assert len(documents) > 0
    assert all(hasattr(doc, 'page_content') for doc in documents)
    assert all(hasattr(doc, 'metadata') for doc in documents)

@pytest.mark.asyncio
async def test_ingestion_service(sample_docs):
    """Test ingestion service"""
    service = IngestionService()
    
    result = await service.ingest_documents(
        file_paths=sample_docs,
        collection_name="test_collection",
        batch_size=10
    )
    
    assert result["success"] is True
    assert result["documents_processed"] == 2
    assert result["chunks_created"] > 0
    assert result["processing_time_seconds"] > 0

@pytest.mark.asyncio
async def test_invalid_files():
    """Test handling of invalid files"""
    service = IngestionService()
    
    result = await service.ingest_documents(
        file_paths=["nonexistent.txt", "invalid.xyz"],
        collection_name="test_collection"
    )
    
    assert result["success"] is False
    assert result["documents_processed"] == 0
    assert len(result["errors"]) > 0
