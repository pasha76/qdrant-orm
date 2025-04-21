"""
Tests for weighted multi-vector search functionality in Qdrant ORM
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


# Define test model with multiple vector fields
class TestProduct(Base):
    """Test product model with multiple vector fields"""
    
    __collection__ = "test_products"
    
    id = Field(String, primary_key=True)
    name = Field(String)
    category = Field(String)
    price = Field(Float, default=0.0)
    image_embedding = VectorField(dimensions=4)  # Small dimension for testing
    text_embedding = VectorField(dimensions=3)   # Small dimension for testing


class MockPoint:
    """Mock Qdrant point for testing"""
    
    def __init__(self, id, payload, vector=None, score=None):
        self.id = id
        self.payload = payload
        self.vector = vector
        if score is not None:
            self.score = score


@patch('qdrant_client.QdrantClient')
class TestWeightedVectorSearch(unittest.TestCase):
    """Test case for weighted vector search functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_client = MagicMock()
        
        # Create a patched engine that uses our mock client
        with patch('qdrant_orm.engine.QdrantClient', return_value=self.mock_client):
            self.engine = QdrantEngine(url="localhost", port=6333)
            self.session = QdrantSession(self.engine)
    
    def test_combined_vector_search(self, mock_qdrant):
        """Test combined vector search with weights"""
        # Setup mock responses for image search
        image_results = [
            MockPoint(
                id="prod1",
                payload={
                    "name": "Product 1", 
                    "category": "electronics", 
                    "price": 10.0
                },
                vector={
                    "image_embedding": [0.1, 0.2, 0.3, 0.4],
                    "text_embedding": [0.5, 0.6, 0.7]
                },
                score=0.95
            ),
            MockPoint(
                id="prod2",
                payload={
                    "name": "Product 2", 
                    "category": "clothing", 
                    "price": 20.0
                },
                vector={
                    "image_embedding": [0.2, 0.3, 0.4, 0.5],
                    "text_embedding": [0.6, 0.7, 0.8]
                },
                score=0.85
            ),
            MockPoint(
                id="prod3",
                payload={
                    "name": "Product 3", 
                    "category": "home", 
                    "price": 30.0
                },
                vector={
                    "image_embedding": [0.3, 0.4, 0.5, 0.6],
                    "text_embedding": [0.7, 0.8, 0.9]
                },
                score=0.75
            )
        ]
        
        # Setup mock responses for text search
        text_results = [
            MockPoint(
                id="prod3",
                payload={
                    "name": "Product 3", 
                    "category": "home", 
                    "price": 30.0
                },
                vector={
                    "image_embedding": [0.3, 0.4, 0.5, 0.6],
                    "text_embedding": [0.7, 0.8, 0.9]
                },
                score=0.92
            ),
            MockPoint(
                id="prod4",
                payload={
                    "name": "Product 4", 
                    "category": "books", 
                    "price": 40.0
                },
                vector={
                    "image_embedding": [0.4, 0.5, 0.6, 0.7],
                    "text_embedding": [0.8, 0.9, 1.0]
                },
                score=0.88
            ),
            MockPoint(
                id="prod1",
                payload={
                    "name": "Product 1", 
                    "category": "electronics", 
                    "price": 10.0
                },
                vector={
                    "image_embedding": [0.1, 0.2, 0.3, 0.4],
                    "text_embedding": [0.5, 0.6, 0.7]
                },
                score=0.65
            )
        ]
        
        # Configure mock to return different results based on which vector field is queried
        def mock_search(**kwargs):
            query_vector = kwargs.get('query_vector')
            if isinstance(query_vector, tuple) and query_vector[0] == 'image_embedding':
                return image_results
            elif isinstance(query_vector, tuple) and query_vector[0] == 'text_embedding':
                return text_results
            return []
        
        self.mock_client.search.side_effect = mock_search
        
        # Create query vectors
        query_image_vector = [0.1, 0.2, 0.3, 0.4]
        query_text_vector = [0.5, 0.6, 0.7]
        
        # Test combined vector search with weights (70% image, 30% text)
        results = self.session.query(TestProduct).combined_vector_search(
            vector_fields_with_weights={
                TestProduct.image_embedding: 0.7,
                TestProduct.text_embedding: 0.3
            },
            query_vectors={
                "image_embedding": query_image_vector,
                "text_embedding": query_text_vector
            },
            limit=3
        ).all()
        
        # Verify search was called for each vector field
        self.assertEqual(self.mock_client.search.call_count, 2)
        
        # Verify results
        self.assertEqual(len(results), 3)
        
        # Expected order based on weighted scores:
        # prod1: 0.95*0.7 + 0.65*0.3 = 0.665 + 0.195 = 0.86
        # prod3: 0.75*0.7 + 0.92*0.3 = 0.525 + 0.276 = 0.801
        # prod2: 0.85*0.7 + 0*0.3 = 0.595
        # prod4: 0*0.7 + 0.88*0.3 = 0.264
        
        # Check that the results are in the expected order
        self.assertEqual(results[0].id, "prod1")
        self.assertEqual(results[1].id, "prod3")
        
        # The third result could be either prod2 or prod4 depending on implementation details
        self.assertIn(results[2].id, ["prod2", "prod4"])
    
    def test_combined_search_with_filters(self, mock_qdrant):
        """Test combined vector search with additional filters"""
        # Setup mock responses
        filtered_results = [
            MockPoint(
                id="prod1",
                payload={
                    "name": "Product 1", 
                    "category": "electronics", 
                    "price": 10.0
                },
                vector={
                    "image_embedding": [0.1, 0.2, 0.3, 0.4],
                    "text_embedding": [0.5, 0.6, 0.7]
                },
                score=0.9
            )
        ]
        
        # Configure mock to return filtered results
        self.mock_client.search.return_value = filtered_results
        
        # Create query vectors
        query_image_vector = [0.1, 0.2, 0.3, 0.4]
        query_text_vector = [0.5, 0.6, 0.7]
        
        # Test combined vector search with filters
        results = self.session.query(TestProduct).filter(
            TestProduct.category == "electronics",
            TestProduct.price < 20.0
        ).combined_vector_search(
            vector_fields_with_weights={
                TestProduct.image_embedding: 0.6,
                TestProduct.text_embedding: 0.4
            },
            query_vectors={
                "image_embedding": query_image_vector,
                "text_embedding": query_text_vector
            },
            limit=2
        ).all()
        
        # Verify search was called with filters
        self.assertEqual(self.mock_client.search.call_count, 2)
        
        # Check that filter was passed to search
        for call_args in self.mock_client.search.call_args_list:
            kwargs = call_args[1]
            self.assertIn('filter', kwargs)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "prod1")
        self.assertEqual(results[0].category, "electronics")
        self.assertEqual(results[0].price, 10.0)
    
    def test_error_handling(self, mock_qdrant):
        """Test error handling in combined vector search"""
        # Create query vectors
        query_image_vector = [0.1, 0.2, 0.3, 0.4]
        
        # Test with missing query vector
        with self.assertRaises(ValueError):
            self.session.query(TestProduct).combined_vector_search(
                vector_fields_with_weights={
                    TestProduct.image_embedding: 0.7,
                    TestProduct.text_embedding: 0.3
                },
                query_vectors={
                    "image_embedding": query_image_vector,
                    # Missing text_embedding
                },
                limit=3
            ).all()


if __name__ == "__main__":
    unittest.main()
