import unittest
from unittest.mock import MagicMock, patch

from qdrant_orm.base import Base, Field, VectorField
from qdrant_orm.types import String, Integer, Vector
from qdrant_orm.query import Query


class Document(Base):
    __collection__ = "documents"
    
    id = Field(field_type=String(), primary_key=True)
    title = Field(field_type=String())
    content = Field(field_type=String())
    embedding = VectorField(dimensions=128)
    image_embedding = VectorField(dimensions=256)


class TestVectorNameParameterFix(unittest.TestCase):
    
    def setUp(self):
        self.mock_session = MagicMock()
        self.mock_client = MagicMock()
        self.mock_session.client = self.mock_client
        
        # Create a sample query vector
        self.query_vector = [0.1] * 128
        self.image_query_vector = [0.2] * 256
    
    def test_default_vector_search(self):
        """Test vector search with default embedding field"""
        query = Query(self.mock_session, Document)
        query.vector_search(Document.embedding, self.query_vector)
        
        # Execute the query
        query.all()
        
        # Check that search was called with correct parameters
        self.mock_client.search.assert_called_once()
        call_args = self.mock_client.search.call_args[1]
        
        # Verify parameters
        self.assertEqual(call_args["collection_name"], "documents")
        self.assertEqual(call_args["query_vector"], self.query_vector)
        self.assertTrue("with_payload" in call_args)
        self.assertTrue("with_vectors" in call_args)
        
        # Verify that named_vector is NOT in the parameters
        self.assertNotIn("named_vector", call_args)
        self.assertNotIn("vector_name", call_args)
    
    def test_custom_vector_search(self):
        """Test vector search with custom vector field"""
        query = Query(self.mock_session, Document)
        query.vector_search(Document.image_embedding, self.image_query_vector)
        
        # Execute the query
        query.all()
        
        # Check that search was called with correct parameters
        self.mock_client.search.assert_called_once()
        call_args = self.mock_client.search.call_args[1]
        
        # Verify parameters
        self.assertEqual(call_args["collection_name"], "documents")
        self.assertEqual(call_args["query_vector"], self.image_query_vector)
        self.assertTrue("with_payload" in call_args)
        self.assertTrue("with_vectors" in call_args)
        
        # Verify that named_vector is NOT in the parameters
        self.assertNotIn("named_vector", call_args)
        self.assertNotIn("vector_name", call_args)
    
    def test_combined_vector_search(self):
        """Test combined vector search with multiple fields"""
        query = Query(self.mock_session, Document)
        query.combined_vector_search(
            vector_fields_with_weights={
                Document.embedding: 0.7,
                Document.image_embedding: 0.3
            },
            query_vectors={
                "embedding": self.query_vector,
                "image_embedding": self.image_query_vector
            }
        )
        
        # Mock search results for each field
        mock_point1 = MagicMock()
        mock_point1.id = "doc1"
        mock_point1.score = 0.9
        mock_point1.payload = {"id": "doc1", "title": "Test Doc 1", "content": "Content 1"}
        # Properly mock the vector data structure
        mock_point1.vector = {"embedding": self.query_vector, "image_embedding": self.image_query_vector}
        
        mock_point2 = MagicMock()
        mock_point2.id = "doc2"
        mock_point2.score = 0.8
        mock_point2.payload = {"id": "doc2", "title": "Test Doc 2", "content": "Content 2"}
        # Properly mock the vector data structure
        mock_point2.vector = {"embedding": self.query_vector, "image_embedding": self.image_query_vector}
        
        self.mock_client.search.side_effect = [
            [mock_point1, mock_point2],  # Results for embedding
            [mock_point2, mock_point1]   # Results for image_embedding
        ]
        
        # Mock the _point_to_model method to avoid conversion issues in testing
        with patch.object(Query, '_point_to_model', return_value=MagicMock()):
            # Execute the query
            query.all()
        
        # Check that search was called twice (once for each field)
        self.assertEqual(self.mock_client.search.call_count, 2)
        
        # Check first call (embedding)
        first_call_args = self.mock_client.search.call_args_list[0][1]
        self.assertEqual(first_call_args["collection_name"], "documents")
        self.assertEqual(first_call_args["query_vector"], self.query_vector)
        
        # Check second call (image_embedding)
        second_call_args = self.mock_client.search.call_args_list[1][1]
        self.assertEqual(second_call_args["collection_name"], "documents")
        self.assertEqual(second_call_args["query_vector"], self.image_query_vector)
        
        # Verify that named_vector is NOT in any of the calls
        self.assertNotIn("named_vector", first_call_args)
        self.assertNotIn("vector_name", first_call_args)
        self.assertNotIn("named_vector", second_call_args)
        self.assertNotIn("vector_name", second_call_args)


if __name__ == "__main__":
    unittest.main()
