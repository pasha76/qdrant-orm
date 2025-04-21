"""
Test for the fixed filter parameter handling in Qdrant ORM
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


class TestFilterParameterHandling(unittest.TestCase):
    """Test case for filter parameter handling fix"""
    
    def test_scroll_filter_parameter(self):
        """Test that scroll method uses scroll_filter parameter instead of filter"""
        # Create a mock QdrantClient
        mock_client = MagicMock()
        # Set up the mock to return empty results
        mock_client.scroll.return_value = ([], None)
        
        # Create a patched QdrantEngine that uses our mock client
        with patch('qdrant_orm.engine.QdrantClient', return_value=mock_client):
            engine = QdrantEngine(url="localhost", port=6333)
            session = QdrantSession(engine)
            
            # Create a query with filters
            query = session.query(TestDocument).filter(
                TestDocument.title == "Test Document"
            )
            
            # Execute the query
            query.all()
            
            # Verify scroll was called with correct parameters
            mock_client.scroll.assert_called_once()
            call_args = mock_client.scroll.call_args[1]
            
            # Check that scroll_filter parameter is used instead of filter
            self.assertIn("scroll_filter", call_args)
            self.assertNotIn("filter", call_args)
            
            # Check that the filter condition is correctly passed
            scroll_filter = call_args["scroll_filter"]
            self.assertEqual(len(scroll_filter.must), 1)
            condition = scroll_filter.must[0]
            self.assertEqual(condition.key, "title")
            self.assertEqual(condition.match.value, "Test Document")


if __name__ == "__main__":
    unittest.main()
