#!/usr/bin/env python3
"""
Script to process 1 million documents with the RAG system
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.massive_ingestion_service import MassiveIngestionService
from app.services.advanced_query_service import AdvancedQueryService
from app.utils.dataset_generator import LargeDatasetGenerator
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def main():
    """Main function to process 1 million documents"""
    print("🚀 Starting Million Document RAG Processing")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = Path("large_dataset")
    if not dataset_path.exists():
        print("📊 Generating 1 million document dataset...")
        generator = LargeDatasetGenerator()
        result = await generator.generate_million_documents(1000000)
        print(f"✅ Generated {result['total_documents']:,} documents in {result['processing_time']:.2f} seconds")
    else:
        print("📁 Dataset already exists, skipping generation")
    
    # Initialize services
    print("\n🔧 Initializing services...")
    ingestion_service = MassiveIngestionService()
    query_service = AdvancedQueryService()
    
    # Process documents
    print("\n📚 Processing 1 million documents...")
    start_time = time.time()
    
    result = await ingestion_service.process_million_documents(
        collection_name="million_docs",
        batch_size=1000,
        max_workers=10
    )
    
    processing_time = time.time() - start_time
    
    print(f"\n✅ Processing completed!")
    print(f"�� Documents processed: {result['total_documents_processed']:,}")
    print(f"📄 Chunks created: {result['total_chunks_created']:,}")
    print(f"⏱️  Total time: {processing_time:.2f} seconds")
    print(f"🚀 Documents/second: {result['documents_per_second']:.2f}")
    print(f"📈 Chunks/second: {result['chunks_per_second']:.2f}")
    print(f"❌ Errors: {result['total_errors']}")
    print(f"📊 Error rate: {result['error_rate']:.2f}%")
    
    # Test queries
    print("\n🔍 Testing advanced queries...")
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain the different types of neural networks",
        "What are the best practices for data science?",
        "How do you implement a recommendation system?",
        "What is the difference between supervised and unsupervised learning?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        try:
            result = await query_service.answer_question_advanced(
                question=query,
                collection_name="million_docs",
                top_k=5,
                use_reranking=True,
                use_query_expansion=True,
                use_hybrid_search=True
            )
            
            print(f"✅ Answer: {result['answer'][:200]}...")
            print(f"📊 Confidence: {result['confidence_score']:.2f}")
            print(f"📚 Sources: {len(result['sources'])}")
            print(f"⏱️  Time: {result['processing_time_seconds']:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print("\n🎉 Million document processing completed successfully!")
    print("🌐 Your RAG system is now ready to handle enterprise-scale queries!")

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("OPENAI_API_KEY", "your-openai-key-here")
    os.environ.setdefault("VECTOR_DB_PROVIDER", "chroma")
    os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    print("⚠️  Please set your OPENAI_API_KEY environment variable before running!")
    print("   export OPENAI_API_KEY='your-actual-openai-key'")
    print()
    
    asyncio.run(main())
