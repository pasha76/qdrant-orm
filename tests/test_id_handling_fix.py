"""
Test for the fixed ID handling in Qdrant ORM
"""
import unittest
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

# Add parent directory to path to import qdrant_orm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_orm import (
    Base, Field, VectorField, 
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector
)
from qdrant_orm.engine import _convert_id_for_qdrant


# Define test model
class TestDocument(Base):
    """Test document model"""
    
    __collection__ = "test_documents"
    
    id = Field(String, primary_key=True)
    title = Field(String)
    content = Field(String)
    embedding = VectorField(dimensions=4)  # Small dimension for testing


class TestIDHandling(unittest.TestCase):
    """Test case for ID handling fix"""
    
    def test_id_conversion(self):
        """Test ID conversion function"""
        # Test UUID handling
        uuid_obj = uuid.uuid4()
        self.assertEqual(_convert_id_for_qdrant(uuid_obj), str(uuid_obj))
        
        # Test UUID string handling
        uuid_str = str(uuid.uuid4())
        self.assertEqual(_convert_id_for_qdrant(uuid_str), uuid_str)
        
        # Test integer handling
        self.assertEqual(_convert_id_for_qdrant(123), 123)
        
        # Test string ID handling - should convert to deterministic UUID
        string_id = "doc1"
        converted_id = _convert_id_for_qdrant(string_id)
        self.assertTrue(isinstance(converted_id, str))
        # UUID5 is deterministic, so calling it again with the same input should give the same result
        self.assertEqual(converted_id, _convert_id_for_qdrant(string_id))
        
        # Different string IDs should produce different UUIDs
        self.assertNotEqual(_convert_id_for_qdrant("doc1"), _convert_id_for_qdrant("doc2"))
    
    def test_string_id_handling_in_session(self):
        """Test string ID handling in QdrantSession"""
        # Create a mock QdrantClient
        mock_client = MagicMock()
        
        # Create a patched QdrantEngine that uses our mock client
        with patch('qdrant_orm.engine.QdrantClient', return_value=mock_client):
            engine = QdrantEngine(url="localhost", port=6333)
            session = QdrantSession(engine)
            
            # Create document with string ID
            doc = TestDocument(
                id="doc1",
                title="Test Document",
                content="This is a test document.",
                embedding=[0.1, 0.2, 0.3, 0.4]
            )
            
            # Add to session and commit
            session.add(doc)
            session.commit()
            
            # Verify upsert was called
            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args[1]
            self.assertEqual(call_args['collection_name'], "test_documents")
            
            # Check that points were passed
            points = call_args['points']
            self.assertEqual(len(points), 1)
            point = points[0]
            
            # ID should not be "doc1" but a converted UUID
            self.assertNotEqual(point.id, "doc1")
            
            # Original ID should be preserved in payload
            self.assertEqual(point.payload['id'], "doc1")
            
            # Test ID mapping is maintained
            self.assertIn(("test_documents", "doc1"), session._id_mapping)
            self.assertEqual(session._id_mapping[("test_documents", "doc1")], point.id)


if __name__ == "__main__":
    unittest.main()
