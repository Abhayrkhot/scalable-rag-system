#!/usr/bin/env python3
"""
Embedding migration script for RAG system
"""
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.embedding_versioning import EmbeddingVersionManager
from app.core.config import settings
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
    parser = argparse.ArgumentParser(description="Migrate embedding models")
    parser.add_argument("--collection", required=True, help="Collection name to migrate")
    parser.add_argument("--new-model", required=True, help="New embedding model name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true", help="Show migration plan without executing")
    parser.add_argument("--force", action="store_true", help="Force migration even if incompatible")
    
    args = parser.parse_args()
    
    # Initialize version manager
    version_manager = EmbeddingVersionManager()
    
    try:
        print(f"🔍 Analyzing migration from {settings.embedding_model} to {args.new_model}")
        print(f"Collection: {args.collection}")
        print("=" * 60)
        
        # Get collection model info
        collection_info = await version_manager.get_collection_model_info(args.collection)
        if "error" in collection_info:
            print(f"❌ Error: {collection_info['error']}")
            return 1
        
        print(f"Current model: {collection_info['current_model']['model_name']}")
        print(f"Dimension: {collection_info['current_model']['dimension']}")
        print(f"Total vectors: {collection_info['total_vectors']}")
        print()
        
        # Validate compatibility
        validation = await version_manager.validate_collection_compatibility(
            args.collection, args.new_model
        )
        
        print("🔍 Compatibility Check:")
        print(f"Compatible: {validation['compatible']}")
        print(f"Reason: {validation['reason']}")
        print(f"Action: {validation['action']}")
        print()
        
        if not validation["compatible"] and not args.force:
            print("❌ Migration cannot proceed due to compatibility issues")
            print("Use --force to override this check")
            return 1
        
        # Create migration plan
        plan = await version_manager.create_migration_plan(args.collection, args.new_model)
        
        print("📋 Migration Plan:")
        print(f"New model: {plan['new_model']}")
        print(f"Document count: {plan['document_count']}")
        print(f"Estimated time: {plan['estimated_time_minutes']} minutes")
        print(f"Estimated cost: ${plan['estimated_cost']:.2f}")
        print()
        
        if plan['recommendations']:
            print("💡 Recommendations:")
            for rec in plan['recommendations']:
                print(f"  - {rec}")
            print()
        
        if args.dry_run:
            print("✅ Dry run completed. Use --force to execute migration.")
            return 0
        
        # Confirm migration
        if not args.force:
            confirm = input("Do you want to proceed with the migration? (y/N): ")
            if confirm.lower() != 'y':
                print("Migration cancelled.")
                return 0
        
        # Execute migration
        print("�� Starting migration...")
        result = await version_manager.migrate_collection(
            args.collection, args.new_model, args.batch_size
        )
        
        if result["success"]:
            print("✅ Migration completed successfully!")
            print(f"New collection: {result['new_collection_name']}")
            print(f"Migrated documents: {result['migrated_documents']}")
            print()
            print("Next steps:")
            print("1. Test the new collection with sample queries")
            print("2. Update your application to use the new collection")
            print("3. Archive or delete the old collection when ready")
        else:
            print(f"❌ Migration failed: {result['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration script error: {e}")
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("OPENAI_API_KEY", "your-openai-key-here")
    os.environ.setdefault("VECTOR_DB_PROVIDER", "chroma")
    os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    print("⚠️  Please set your OPENAI_API_KEY environment variable before running!")
    print("   export OPENAI_API_KEY='your-actual-openai-key'")
    print()
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
