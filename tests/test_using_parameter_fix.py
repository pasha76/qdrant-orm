"""
Test file to verify the fix for the 'using' parameter issue in vector search.

This test ensures that vector searches work correctly without using the
'using', 'vector_name', or 'named_vector' parameters that are not supported
in older versions of the Qdrant client.
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from qdrant_orm import (
    Base, Field, VectorField, 
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector
)


# Define a model with multiple vector fields for testing
class Document(Base):
    """Document model with multiple vector fields for testing"""
    
    __collection__ = "documents"
    
    id = Field(String, primary_key=True)
    title = Field(String)
    content = Field(String)
    rating = Field(Float)
    # Multiple vector fields
    image_embedding = VectorField(dimensions=512)
    text_embedding = VectorField(dimensions=384)


class TestUsingParameterFix(unittest.TestCase):
    """Test case for the 'using' parameter fix in vector search"""
    
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
        self.mock_search_result.payload = {"id": "doc1", "title": "Test Document", "content": "Test content", "rating": 4.5}
        
        # Mock the search method to return our mock result
        self.mock_client.search.return_value = [self.mock_search_result]
        
        # Mock the point_to_model method
        self.session._point_to_model = MagicMock(return_value=Document(
            id="doc1", 
            title="Test Document", 
            content="Test content", 
            rating=4.5
        ))
        
        # Create test vectors
        self.test_image_vector = np.random.rand(512).tolist()
        self.test_text_vector = np.random.rand(384).tolist()

    def test_single_vector_search_no_using_parameter(self):
        """Test that single vector search works without the 'using' parameter"""
        # Perform a vector search
        results = self.session.query(Document).vector_search(
            Document.image_embedding,
            vector=self.test_image_vector
        ).all()
        
        # Verify the search was called with the correct parameters
        self.mock_client.search.assert_called_once()
        
        # Get the kwargs that were passed to the search method
        call_kwargs = self.mock_client.search.call_args[1]
        
        # Verify that 'using', 'vector_name', and 'named_vector' parameters were not used
        self.assertNotIn('using', call_kwargs)
        self.assertNotIn('vector_name', call_kwargs)
        self.assertNotIn('named_vector', call_kwargs)
        
        # Verify that the query_vector parameter was used correctly
        self.assertEqual(call_kwargs['query_vector'], self.test_image_vector)
        
        # Verify that we got results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "doc1")

    def test_combined_vector_search_no_using_parameter(self):
        """Test that combined vector search works without the 'using' parameter"""
        # Mock the search and retrieve methods for combined search
        self.mock_client.search.return_value = [self.mock_search_result]
        self.mock_client.retrieve.return_value = [self.mock_search_result]
        
        # Perform a combined vector search
        results = self.session.query(Document).combined_vector_search(
            vector_fields_with_weights={
                Document.image_embedding: 0.7,
                Document.text_embedding: 0.3
            },
            query_vectors={
                "image_embedding": self.test_image_vector,
                "text_embedding": self.test_text_vector
            },
            limit=5
        ).all()
        
        # Verify the search was called
        self.assertTrue(self.mock_client.search.called)
        
        # Check all calls to search to ensure none used the unsupported parameters
        for call in self.mock_client.search.call_args_list:
            call_kwargs = call[1]
            self.assertNotIn('using', call_kwargs)
            self.assertNotIn('vector_name', call_kwargs)
            self.assertNotIn('named_vector', call_kwargs)
        
        # Verify that we got results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "doc1")


if __name__ == '__main__':
    unittest.main()
