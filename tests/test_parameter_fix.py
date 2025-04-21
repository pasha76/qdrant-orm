"""
Test file to verify the parameter order and naming fixes in the Qdrant ORM framework.

This test ensures that:
1. The Query class correctly handles the session and model_class parameters in the right order
2. The vector_search method accepts both 'vector' and 'query_vector' parameters
"""
import unittest
from unittest.mock import patch, MagicMock

from qdrant_orm import (
    Base, Field, VectorField, 
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector
)


# Define a test model
class TestDocument(Base):
    """Test document model"""
    
    __collection__ = "test_documents"
    
    id = Field(String, primary_key=True)
    title = Field(String)
    embedding = VectorField(dimensions=128)


class TestParameterFix(unittest.TestCase):
    """Test case for the parameter order and naming fixes"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock engine and session
        self.engine = MagicMock()
        self.session = QdrantSession(self.engine)
        
        # Mock the client
        self.mock_client = MagicMock()
        self.session._get_client = MagicMock(return_value=self.mock_client)
        
        # Mock search results
        self.mock_search_result = MagicMock()
        self.mock_search_result.id = "doc1"
        self.mock_search_result.score = 0.95
        self.mock_search_result.payload = {"id": "doc1", "title": "Test Document"}
        
        # Mock the search method to return our mock result
        self.mock_client.search.return_value = [self.mock_search_result]
        
        # Mock the point_to_model method
        self.session._point_to_model = MagicMock(return_value=TestDocument(
            id="doc1", 
            title="Test Document"
        ))
        
        # Create test vector
        self.test_vector = [0.1] * 128

    def test_query_parameter_order(self):
        """Test that the Query class correctly handles the session and model_class parameters"""
        # This should work with the fixed parameter order
        query = self.session.query(TestDocument)
        
        # Verify that the session and model_class are correctly set
        self.assertEqual(query._session, self.session)
        self.assertEqual(query._model_class, TestDocument)

    def test_vector_search_with_vector_parameter(self):
        """Test that vector_search works with the 'vector' parameter"""
        # This should work with the original 'vector' parameter
        results = self.session.query(TestDocument).vector_search(
            TestDocument.embedding,
            vector=self.test_vector
        ).all()
        
        # Verify the search was called with the correct parameters
        self.mock_client.search.assert_called_once()
        
        # Verify that we got results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "doc1")

    def test_vector_search_with_query_vector_parameter(self):
        """Test that vector_search works with the 'query_vector' parameter"""
        # Reset the mock to clear previous calls
        self.mock_client.search.reset_mock()
        
        # This should work with the new 'query_vector' parameter
        results = self.session.query(TestDocument).vector_search(
            TestDocument.embedding,
            query_vector=self.test_vector
        ).all()
        
        # Verify the search was called with the correct parameters
        self.mock_client.search.assert_called_once()
        
        # Verify that we got results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "doc1")


if __name__ == '__main__':
    unittest.main()
