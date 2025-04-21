"""
Unit tests for Qdrant ORM framework
"""
import unittest
import numpy as np
import os
import sys

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


class TestQdrantORM(unittest.TestCase):
    """Test case for Qdrant ORM"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = QdrantEngine(url="localhost", port=6333)
        self.session = QdrantSession(self.engine)
        
        # Create collection
        Base.metadata.create_all(self.engine)
    
    def tearDown(self):
        """Clean up after tests"""
        Base.metadata.drop_all(self.engine)
    
    def test_model_creation(self):
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
    
    def test_crud_operations(self):
        """Test basic CRUD operations"""
        # Create
        doc = TestDocument(
            id="test1",
            title="Test Document",
            content="Test content",
            embedding=[0.1, 0.2, 0.3, 0.4]
        )
        self.session.add(doc)
        self.session.commit()
        
        # Read
        retrieved_doc = self.session.get(TestDocument, "test1")
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc.id, "test1")
        self.assertEqual(retrieved_doc.title, "Test Document")
        
        # Update
        retrieved_doc.title = "Updated Title"
        self.session.add(retrieved_doc)
        self.session.commit()
        
        # Verify update
        updated_doc = self.session.get(TestDocument, "test1")
        self.assertEqual(updated_doc.title, "Updated Title")
        
        # Delete
        self.session.delete(updated_doc)
        self.session.commit()
        
        # Verify deletion
        deleted_doc = self.session.get(TestDocument, "test1")
        self.assertIsNone(deleted_doc)
    
    def test_query_interface(self):
        """Test query interface"""
        # Create test data
        docs = []
        for i in range(5):
            doc = TestDocument(
                id=f"test{i}",
                title=f"Test Document {i}",
                content=f"Content {i}",
                score=float(i),
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            )
            docs.append(doc)
        
        CRUDOperations.bulk_insert(self.session, docs)
        
        # Test filter
        results = self.session.query(TestDocument).filter(
            TestDocument.score > 2.0
        ).all()
        self.assertEqual(len(results), 2)  # test3, test4
        
        # Test order by
        results = self.session.query(TestDocument).order_by(
            TestDocument.score.desc()
        ).all()
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0].id, "test4")  # Highest score
        
        # Test limit
        results = self.session.query(TestDocument).limit(2).all()
        self.assertEqual(len(results), 2)
        
        # Test count
        count = self.session.query(TestDocument).count()
        self.assertEqual(count, 5)
        
        filtered_count = self.session.query(TestDocument).filter(
            TestDocument.score < 3.0
        ).count()
        self.assertEqual(filtered_count, 3)  # test0, test1, test2
    
    def test_vector_search(self):
        """Test vector search"""
        # Create test data with specific embeddings
        docs = []
        for i in range(5):
            doc = TestDocument(
                id=f"test{i}",
                title=f"Test Document {i}",
                content=f"Content {i}",
                score=float(i),
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            )
            docs.append(doc)
        
        CRUDOperations.bulk_insert(self.session, docs)
        
        # Search for vectors similar to test4
        query_vector = [0.4, 0.8, 1.2, 1.6]  # Same as test4's embedding
        results = self.session.query(TestDocument).vector_search(
            TestDocument.embedding,
            query_vector=query_vector,
            limit=2
        ).all()
        
        self.assertEqual(len(results), 2)
        # The most similar should be test4 itself
        self.assertEqual(results[0].id, "test4")
    
    def test_advanced_crud(self):
        """Test advanced CRUD operations"""
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
        
        # Test update_or_create
        doc, created = CRUDOperations.update_or_create(
            self.session,
            TestDocument,
            defaults={"content": "Updated via update_or_create", "score": 10.0},
            id="new_doc",
            title="New Document"
        )
        
        self.assertFalse(created)
        self.assertEqual(doc.content, "Updated via update_or_create")
        self.assertEqual(doc.score, 10.0)
        
        # Test bulk operations
        docs = []
        for i in range(3):
            doc = TestDocument(
                id=f"bulk{i}",
                title=f"Bulk Document {i}",
                content=f"Bulk Content {i}",
                embedding=[0.1, 0.2, 0.3, 0.4]
            )
            docs.append(doc)
        
        CRUDOperations.bulk_insert(self.session, docs)
        
        # Verify bulk insert
        count = self.session.query(TestDocument).filter(
            TestDocument.id.in_(["bulk0", "bulk1", "bulk2"])
        ).count()
        self.assertEqual(count, 3)
        
        # Test delete_by_filter
        CRUDOperations.delete_by_filter(
            self.session,
            TestDocument,
            TestDocument.id.in_(["bulk0", "bulk1"])
        )
        
        # Verify deletion
        count = self.session.query(TestDocument).filter(
            TestDocument.id.in_(["bulk0", "bulk1", "bulk2"])
        ).count()
        self.assertEqual(count, 1)  # Only bulk2 should remain


if __name__ == "__main__":
    unittest.main()
