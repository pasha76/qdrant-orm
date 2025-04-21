"""
Mock-based tests for Qdrant ORM framework
"""
import unittest
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import qdrant_orm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_orm import (
    Base, Field, VectorField, 
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector
)
from qdrant_orm.crud import CRUDOperations


# Define test model
class TestDocument(Base):
    """Test document model"""
    
    __collection__ = "test_documents"
    
    id = Field(String, primary_key=True)
    title = Field(String)
    content = Field(String, nullable=True)
    score = Field(Float, default=0.0)
    is_active = Field(Boolean, default=True)
    embedding = VectorField(dimensions=4)  # Small dimension for testing


class MockPoint:
    """Mock Qdrant point for testing"""
    
    def __init__(self, id, payload, vector):
        self.id = id
        self.payload = payload
        self.vector = vector


@patch('qdrant_client.QdrantClient')
class TestQdrantORM(unittest.TestCase):
    """Test case for Qdrant ORM with mocked Qdrant client"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_client = MagicMock()
        
        # Create a patched engine that uses our mock client
        with patch('qdrant_orm.engine.QdrantClient', return_value=self.mock_client):
            self.engine = QdrantEngine(url="localhost", port=6333)
            self.session = QdrantSession(self.engine)
    
    def test_model_creation(self, mock_qdrant):
        """Test model creation"""
        doc = TestDocument(
            id="test1",
            title="Test Document",
            content="Test content",
            embedding=[0.1, 0.2, 0.3, 0.4]
        )
        
        self.assertEqual(doc.id, "test1")
        self.assertEqual(doc.title, "Test Document")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.score, 0.0)  # Default value
        self.assertEqual(doc.is_active, True)  # Default value
        self.assertEqual(doc.embedding, [0.1, 0.2, 0.3, 0.4])
    
    def test_crud_operations(self, mock_qdrant):
        """Test basic CRUD operations"""
        # Setup mock responses
        self.mock_client.retrieve.return_value = [
            MockPoint(
                id="test1",
                payload={"title": "Test Document", "content": "Test content", "score": 0.0, "is_active": True},
                vector=[0.1, 0.2, 0.3, 0.4]
            )
        ]
        
        # Create
        doc = TestDocument(
            id="test1",
            title="Test Document",
            content="Test content",
            embedding=[0.1, 0.2, 0.3, 0.4]
        )
        self.session.add(doc)
        self.session.commit()
        
        # Verify upsert was called
        self.mock_client.upsert.assert_called_once()
        
        # Read
        retrieved_doc = self.session.get(TestDocument, "test1")
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc.id, "test1")
        self.assertEqual(retrieved_doc.title, "Test Document")
        
        # Update mock response for the updated document
        self.mock_client.retrieve.return_value = [
            MockPoint(
                id="test1",
                payload={"title": "Updated Title", "content": "Test content", "score": 0.0, "is_active": True},
                vector=[0.1, 0.2, 0.3, 0.4]
            )
        ]
        
        # Update
        retrieved_doc.title = "Updated Title"
        self.session.add(retrieved_doc)
        self.session.commit()
        
        # Verify upsert was called again
        self.assertEqual(self.mock_client.upsert.call_count, 2)
        
        # Verify update
        updated_doc = self.session.get(TestDocument, "test1")
        self.assertEqual(updated_doc.title, "Updated Title")
        
        # Setup mock for deletion verification
        self.mock_client.retrieve.return_value = []
        
        # Delete
        self.session.delete(updated_doc)
        self.session.commit()
        
        # Verify delete was called
        self.mock_client.delete.assert_called_once()
        
        # Verify deletion
        deleted_doc = self.session.get(TestDocument, "test1")
        self.assertIsNone(deleted_doc)
    
    def test_query_interface(self, mock_qdrant):
        """Test query interface"""
        # Setup mock responses for scroll
        self.mock_client.scroll.return_value = (
            [
                MockPoint(
                    id=f"test{i}",
                    payload={
                        "title": f"Test Document {i}", 
                        "content": f"Content {i}", 
                        "score": float(i), 
                        "is_active": True
                    },
                    vector=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
                )
                for i in range(5)
            ],
            None  # next_page_offset
        )
        
        # Setup mock response for count
        count_result = MagicMock()
        count_result.count = 5
        self.mock_client.count.return_value = count_result
        
        # Test filter
        results = self.session.query(TestDocument).filter(
            TestDocument.score > 2.0
        ).all()
        
        # Verify scroll was called with filter
        self.mock_client.scroll.assert_called()
        
        # Test count
        count = self.session.query(TestDocument).count()
        self.assertEqual(count, 5)
        
        # Verify count was called
        self.mock_client.count.assert_called()
    
    def test_vector_search(self, mock_qdrant):
        """Test vector search"""
        # Setup mock responses for search
        self.mock_client.search.return_value = [
            MockPoint(
                id="test4",
                payload={
                    "title": "Test Document 4", 
                    "content": "Content 4", 
                    "score": 4.0, 
                    "is_active": True
                },
                vector=[0.4, 0.8, 1.2, 1.6]
            ),
            MockPoint(
                id="test3",
                payload={
                    "title": "Test Document 3", 
                    "content": "Content 3", 
                    "score": 3.0, 
                    "is_active": True
                },
                vector=[0.3, 0.6, 0.9, 1.2]
            )
        ]
        
        # Search for vectors similar to test4
        query_vector = [0.4, 0.8, 1.2, 1.6]
        results = self.session.query(TestDocument).vector_search(
            TestDocument.embedding,
            query_vector=query_vector,
            limit=2
        ).all()
        
        # Verify search was called
        self.mock_client.search.assert_called_once()
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "test4")
    
    def test_advanced_crud(self, mock_qdrant):
        """Test advanced CRUD operations"""
        # Setup mock for get_or_create (first call - not found)
        self.mock_client.scroll.return_value = ([], None)
        
        # Setup mock for after creation
        self.mock_client.retrieve.return_value = [
            MockPoint(
                id="new_doc",
                payload={
                    "title": "New Document", 
                    "content": "New content", 
                    "score": 0.0, 
                    "is_active": True
                },
                vector=[0.1, 0.2, 0.3, 0.4]
            )
        ]
        
        # Test get_or_create
        doc, created = CRUDOperations.get_or_create(
            self.session,
            TestDocument,
            defaults={"content": "New content", "embedding": [0.1, 0.2, 0.3, 0.4]},
            id="new_doc",
            title="New Document"
        )
        
        self.assertTrue(created)
        self.assertEqual(doc.id, "new_doc")
        self.assertEqual(doc.title, "New Document")
        
        # Setup mock for get_or_create (second call - found)
        self.mock_client.scroll.return_value = (
            [
                MockPoint(
                    id="new_doc",
                    payload={
                        "title": "New Document", 
                        "content": "New content", 
                        "score": 0.0, 
                        "is_active": True
                    },
                    vector=[0.1, 0.2, 0.3, 0.4]
                )
            ],
            None
        )
        
        # Test get_or_create with existing
        doc, created = CRUDOperations.get_or_create(
            self.session,
            TestDocument,
            defaults={"content": "Updated content"},
            id="new_doc",
            title="New Document"
        )
        
        self.assertFalse(created)
        self.assertEqual(doc.id, "new_doc")


if __name__ == "__main__":
    unittest.main()
