import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "version" in data

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_ingest_without_auth():
    """Test ingestion endpoint without authentication"""
    response = client.post("/ingest/", json={
        "file_paths": ["test.txt"],
        "collection_name": "test"
    })
    assert response.status_code == 401

def test_query_without_auth():
    """Test query endpoint without authentication"""
    response = client.post("/query/", json={
        "question": "What is AI?",
        "collection_name": "test"
    })
    assert response.status_code == 401

def test_ingest_with_auth():
    """Test ingestion endpoint with authentication"""
    headers = {"x-api-key": "test-key"}
    response = client.post("/ingest/", 
                          json={
                              "file_paths": ["sample_docs/test1.txt"],
                              "collection_name": "test"
                          },
                          headers=headers)
    # This will fail due to missing OpenAI key, but should not be 401
    assert response.status_code != 401

def test_query_with_auth():
    """Test query endpoint with authentication"""
    headers = {"x-api-key": "test-key"}
    response = client.post("/query/", 
                          json={
                              "question": "What is AI?",
                              "collection_name": "test"
                          },
                          headers=headers)
    # This will fail due to missing OpenAI key, but should not be 401
    assert response.status_code != 401
