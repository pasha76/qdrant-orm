import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path so we can import the qdrant_orm package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_orm.base import Base, Field, VectorField
from qdrant_orm.query import Query


class TestVectorSearchFormat(unittest.TestCase):
    """Test the vector search parameter format fixes"""

    def setUp(self):
        # Create a mock session
        self.mock_session = MagicMock()
        self.mock_client = MagicMock()
        self.mock_session.client = self.mock_client
        
        # Create a test model class
        class TestModel(Base):
            __collection__ = "test_collection"
            
            id = Field(field_type="keyword", primary_key=True)
            name = Field(field_type="keyword")
            embedding = VectorField(dimensions=128)
            image_embedding = VectorField(dimensions=512)
            text_embedding = VectorField(dimensions=256)
        
        self.TestModel = TestModel

    def test_single_vector_search_format(self):
        """Test that single vector search uses the correct parameter format"""
        # Create a query with vector search
        query = Query(self.mock_session, self.TestModel)
        query_vector = [0.1] * 128
        query.vector_search("embedding", query_vector, limit=5)
        
        # Execute the query
        query.all()
        
        # Check that the search method was called with the correct parameters
        self.mock_client.search.assert_called_once()
        call_args = self.mock_client.search.call_args[1]
        
        # Verify the query_vector parameter is passed directly as a list, not a dictionary
        self.assertEqual(call_args["query_vector"], query_vector)
        self.assertNotIn("vector_name", call_args)  # Default field name doesn't need vector_name
        
    def test_single_vector_search_with_custom_field(self):
        """Test that single vector search with a custom field name uses the correct parameter format"""
        # Create a query with vector search on a non-default field
        query = Query(self.mock_session, self.TestModel)
        query_vector = [0.1] * 512
        query.vector_search("image_embedding", query_vector, limit=5)
        
        # Execute the query
        query.all()
        
        # Check that the search method was called with the correct parameters
        self.mock_client.search.assert_called_once()
        call_args = self.mock_client.search.call_args[1]
        
        # Verify the query_vector parameter is passed directly as a list, not a dictionary
        self.assertEqual(call_args["query_vector"], query_vector)
        self.assertEqual(call_args["vector_name"], "image_embedding")  # Custom field name needs vector_name
        
    def test_combined_vector_search_format(self):
        """Test that combined vector search uses the correct parameter format for each field"""
        # Create a query with combined vector search
        query = Query(self.mock_session, self.TestModel)
        image_vector = [0.1] * 512
        text_vector = [0.2] * 256
        
        query.combined_vector_search(
            vector_fields_with_weights={
                "image_embedding": 0.7,
                "text_embedding": 0.3
            },
            query_vectors={
                "image_embedding": image_vector,
                "text_embedding": text_vector
            },
            limit=5
        )
        
        # Execute the query
        query.all()
        
        # Check that the search method was called twice (once for each field)
        self.assertEqual(self.mock_client.search.call_count, 2)
        
        # Get the call arguments for each call
        call_args_list = self.mock_client.search.call_args_list
        
        # Find the calls for each field
        image_call = None
        text_call = None
        
        for call in call_args_list:
            args = call[1]
            if "vector_name" in args and args["vector_name"] == "image_embedding":
                image_call = args
            elif "vector_name" in args and args["vector_name"] == "text_embedding":
                text_call = args
        
        # Verify the image embedding call
        self.assertIsNotNone(image_call)
        self.assertEqual(image_call["query_vector"], image_vector)
        self.assertEqual(image_call["vector_name"], "image_embedding")
        
        # Verify the text embedding call
        self.assertIsNotNone(text_call)
        self.assertEqual(text_call["query_vector"], text_vector)
        self.assertEqual(text_call["vector_name"], "text_embedding")


if __name__ == "__main__":
    unittest.main()
