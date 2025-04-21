"""
Test for the fixed vector field naming in Qdrant ORM
"""
import unittest
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


# Define test model
class TestDocument(Base):
    """Test document model"""
    
    __collection__ = "test_documents"
    
    id = Field(String, primary_key=True)
    title = Field(String)
    content = Field(String)
    embedding = VectorField(dimensions=4)  # Small dimension for testing


class TestVectorFieldNaming(unittest.TestCase):
    """Test case for vector field naming fix"""
    
    def test_vector_search_parameter_format(self):
        """Test that vector search uses correct parameter format"""
        # Create a mock QdrantClient
        mock_client = MagicMock()
        # Set up the mock to return empty results
        mock_client.search.return_value = []
        
        # Create a patched QdrantEngine that uses our mock client
        with patch('qdrant_orm.engine.QdrantClient', return_value=mock_client):
            engine = QdrantEngine(url="localhost", port=6333)
            session = QdrantSession(engine)
            
            # Create a query with vector search
            query = session.query(TestDocument).vector_search(
                TestDocument.embedding, 
                [0.1, 0.2, 0.3, 0.4]
            )
            
            # Execute the query
            query.all()
            
            # Verify search was called with correct parameters
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args[1]
            
            # Check that query_vector is a dictionary with field name as key
            self.assertIn("query_vector", call_args)
            self.assertIsInstance(call_args["query_vector"], dict)
            self.assertIn("embedding", call_args["query_vector"])
            self.assertEqual(call_args["query_vector"]["embedding"], [0.1, 0.2, 0.3, 0.4])
    
    def test_combined_vector_search_parameter_format(self):
        """Test that combined vector search uses correct parameter format"""
        # Create a mock QdrantClient
        mock_client = MagicMock()
        # Set up the mock to return empty results
        mock_client.search.return_value = []
        
        # Create a patched QdrantEngine that uses our mock client
        with patch('qdrant_orm.engine.QdrantClient', return_value=mock_client):
            engine = QdrantEngine(url="localhost", port=6333)
            session = QdrantSession(engine)
            
            # Create a query with combined vector search
            query = session.query(TestDocument).combined_vector_search(
                vector_fields_with_weights={
                    TestDocument.embedding: 1.0
                },
                query_vectors={
                    "embedding": [0.1, 0.2, 0.3, 0.4]
                }
            )
            
            # Execute the query
            query.all()
            
            # Verify search was called with correct parameters
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args[1]
            
            # Check that query_vector is a dictionary with field name as key
            self.assertIn("query_vector", call_args)
            self.assertIsInstance(call_args["query_vector"], dict)
            self.assertIn("embedding", call_args["query_vector"])
            self.assertEqual(call_args["query_vector"]["embedding"], [0.1, 0.2, 0.3, 0.4])


if __name__ == "__main__":
    unittest.main()
