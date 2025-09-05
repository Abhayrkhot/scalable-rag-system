#!/bin/bash

# Run tests for the RAG system
echo "Running RAG System Tests..."

# Set test environment
export API_KEY="test-key"
export OPENAI_API_KEY="test-openai-key"
export VECTOR_DB_PROVIDER="chroma"
export CHROMA_PERSIST_DIRECTORY="./test_data/chroma_db"

# Create test data directory
mkdir -p test_data/chroma_db

# Run pytest
pytest tests/ -v --tb=short

# Cleanup test data
rm -rf test_data/

echo "Tests completed!"
