"""
Test file to verify the _get_client method fix in the QdrantSession class.

This test ensures that the _get_client method is properly implemented
and returns the client attribute of the QdrantSession object.
"""
import unittest
from unittest.mock import patch, MagicMock

from qdrant_orm import (
    Base, Field, VectorField, 
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector
)


class TestGetClientMethodFix(unittest.TestCase):
    """Test case for the _get_client method fix in QdrantSession"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock engine
        self.engine = MagicMock()
        self.mock_client = MagicMock()
        self.engine.get_client.return_value = self.mock_client
        
        # Create a session with the mock engine
        self.session = QdrantSession(self.engine)

    def test_get_client_method_exists(self):
        """Test that the _get_client method exists in QdrantSession"""
        # Verify that the method exists
        self.assertTrue(hasattr(self.session, '_get_client'))
        
        # Verify that it's callable
        self.assertTrue(callable(self.session._get_client))
    
    def test_get_client_returns_client(self):
        """Test that _get_client returns the client attribute"""
        # Verify that _get_client returns the client attribute
        self.assertEqual(self.session._get_client(), self.session.client)
        
        # Verify that the client attribute is the mock client
        self.assertEqual(self.session.client, self.mock_client)
    
    def test_query_uses_get_client(self):
        """Test that the query method works with _get_client"""
        # Define a test model
        class TestModel(Base):
            __collection__ = "test_collection"
            id = Field(String, primary_key=True)
        
        # Mock the Query class
        with patch('qdrant_orm.query.Query') as MockQuery:
            # Create a query
            query = self.session.query(TestModel)
            
            # Verify that Query was called with the session and model
            MockQuery.assert_called_once_with(self.session, TestModel)


if __name__ == '__main__':
    unittest.main()
