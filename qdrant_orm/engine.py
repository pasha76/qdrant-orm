"""
Engine and session classes for Qdrant ORM
"""
from typing import Dict, Any, Type, List, Optional, Union, Tuple
import uuid
import re

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .base import Base


class QdrantEngine:
    """Manages connection to Qdrant server"""
    
    def __init__(self, url: str = "localhost", port: int = 6333, api_key: Optional[str] = None, 
                 https: bool = False, prefix: Optional[str] = None, timeout: float = 5.0):
        """
        Initialize a connection to Qdrant server
        
        Args:
            url: Qdrant server URL
            port: Qdrant server port
            api_key: API key for authentication
            https: Whether to use HTTPS
            prefix: URL prefix
            timeout: Connection timeout in seconds
        """
        self.client = QdrantClient(
            url=url,
            port=port,
            api_key=api_key,
            https=https,
            prefix=prefix,
            timeout=timeout
        )
    
    def create_collection(self, collection_name: str, model_class: Type[Base]):
        """
        Create a collection for the given model class
        
        Args:
            collection_name: Name of the collection
            model_class: Model class to create collection for
        """
        # Extract vector fields
        vector_fields = {}
        for name, field in model_class._fields.items():
            if hasattr(field, 'dimensions'):
                vector_fields[name] = (field.dimensions, field.distance)
        
        if not vector_fields:
            raise ValueError(f"Model {model_class.__name__} has no vector fields")
        
        # Use the first vector field as the primary vector
        primary_vector_name = next(iter(vector_fields))
        primary_dimensions, primary_distance = vector_fields[primary_vector_name]
        
        # Create vector configs for named vectors (if multiple)
        vectors_config = {}
        sparse_vectors_config={}
        if len(vector_fields) > 1:
            for name, (dimensions, distance) in vector_fields.items():
                if dimensions and distance:
                    vectors_config[name] = qmodels.VectorParams(
                        size=dimensions,
                        distance=distance
                    )
                else:
                    sparse_vectors_config[name] = qmodels.SparseVectorParams()
        
        # Prepare schema for payload fields
        payload_schema = {}
        for name, field in model_class._fields.items():
            if not hasattr(field, 'dimensions'):  # Skip vector fields
                payload_schema[name] = field.field_type
        
        # Create collection
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=(
                vectors_config or 
                qmodels.VectorParams(
                    size=primary_dimensions,
                    distance=primary_distance
                )
            ),
            sparse_vectors_config=sparse_vectors_config
        )
    
    def drop_collection(self, collection_name: str):
        """
        Drop a collection
        
        Args:
            collection_name: Name of the collection to drop
        """
        self.client.delete_collection(collection_name=collection_name)
    
    def get_client(self) -> QdrantClient:
        """Get the underlying Qdrant client"""
        return self.client


def _convert_id_for_qdrant(id_value):
    """
    Convert an ID value to a format acceptable by Qdrant (UUID or unsigned integer)
    
    Args:
        id_value: The ID value to convert
        
    Returns:
        UUID or integer suitable for Qdrant
    """
    # If it's already a UUID or UUID string, use it
    if isinstance(id_value, uuid.UUID):
        return str(id_value)
    
    # If it's a UUID string, convert to UUID object
    if isinstance(id_value, str) and re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', id_value.lower()):
        return id_value
    
    # If it's an integer, use it directly
    if isinstance(id_value, int) and id_value >= 0:
        return id_value
    
    # For string IDs, generate a deterministic UUID based on the string
    if isinstance(id_value, str):
        # Use UUID5 with DNS namespace for deterministic generation
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_value))
    
    # For any other type, convert to string and then to UUID
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(id_value)))


class QdrantSession:
    """Manages a session for performing operations"""
    
    def __init__(self, engine: QdrantEngine):
        """
        Initialize a session
        
        Args:
            engine: QdrantEngine instance
        """
        self.engine = engine
        self.client = engine.get_client()
        self._pending = []
        # Store mapping between original IDs and Qdrant IDs
        self._id_mapping = {}
    
    def add(self, instance: Base):
        """
        Add an instance to the session
        
        Args:
            instance: Model instance to add
        """
        self._pending.append(('add', instance))
    
    def delete(self, instance: Base):
        """
        Mark an instance for deletion
        
        Args:
            instance: Model instance to delete
        """
        self._pending.append(('delete', instance))
    
    def commit(self):
        """Commit all pending changes"""
        # Group operations by collection
        operations_by_collection = {}
        
        for op, instance in self._pending:
            collection = instance.__class__.__collection__
            if collection not in operations_by_collection:
                operations_by_collection[collection] = {'add': [], 'delete': []}
            
            operations_by_collection[collection][op].append(instance)
        
        # Process each collection
        for collection, operations in operations_by_collection.items():
            # Process additions
            if operations['add']:
                points = []
                for instance in operations['add']:
                    # Extract vector fields and payload
                    vectors = {}
                    payload = {}
                    
                    for name, value in instance._values.items():
                        field = instance.__class__._fields.get(name)
                        if hasattr(field, 'dimensions'):
                            vectors[name] = value
                        else:
                            payload[name] = value
                    
                    # Generate ID if not provided
                    original_id = getattr(instance, instance.__class__._pk_field, None)
                    if original_id is None:
                        original_id = str(uuid.uuid4())
                        setattr(instance, instance.__class__._pk_field, original_id)
                    
                    # Convert ID to Qdrant-compatible format
                    qdrant_id = _convert_id_for_qdrant(original_id)
                    
                    # Store the mapping between original ID and Qdrant ID
                    self._id_mapping[(collection, original_id)] = qdrant_id
                    
                    # Store original ID in payload for retrieval
                    pk_field = instance.__class__._pk_field
                    payload[pk_field] = original_id
                    
                    # If only one vector field, use it as the primary vector
                    if len(vectors) == 1:
                        vector = next(iter(vectors.values()))
                        points.append(qmodels.PointStruct(
                            id=qdrant_id,
                            vector=vector,
                            payload=payload
                        ))
                    else:
                        # Multiple vector fields
                        points.append(qmodels.PointStruct(
                            id=qdrant_id,
                            vector=vectors,
                            payload=payload
                        ))
                
                # Upsert points
                self.client.upsert(
                    collection_name=collection,
                    points=points
                )
            
            # Process deletions
            if operations['delete']:
                point_ids = []
                for instance in operations['delete']:
                    original_id = instance.pk
                    # Look up the Qdrant ID from mapping or convert it
                    qdrant_id = self._id_mapping.get(
                        (collection, original_id), 
                        _convert_id_for_qdrant(original_id)
                    )
                    point_ids.append(qdrant_id)
                
                self.client.delete(
                    collection_name=collection,
                    points_selector=qmodels.PointIdsList(
                        points=point_ids
                    )
                )
        
        # Clear pending operations
        self._pending.clear()
    
    def query(self, model_class: Type[Base]):
        """
        Create a query for the given model class
        
        Args:
            model_class: Model class to query
            
        Returns:
            Query object
        """
        from .query import Query
        return Query(self, model_class)
    
    def _get_client(self):
        """
        Get the underlying Qdrant client
        
        Returns:
            QdrantClient instance
        """
        return self.client
    
    def _point_to_model(self, point, model_class: Type[Base]):
        """
        Convert a Qdrant point to a model instance
        
        Args:
            point: Qdrant point object
            model_class: Model class to convert to
            
        Returns:
            Model instance
        """
        # Combine payload and vector data
        data = dict(point.payload)
        
        # Handle vector data
        if hasattr(point, 'vector') and point.vector is not None:
            if isinstance(point.vector, dict):
                # Multiple named vectors
                for name, vector in point.vector.items():
                    data[name] = vector
            else:
                # Single vector - find the vector field name
                vector_field_name = None
                for name, field in model_class._fields.items():
                    if hasattr(field, 'dimensions'):
                        vector_field_name = name
                        break
                
                if vector_field_name:
                    data[vector_field_name] = point.vector
        
        # Use the original ID from payload if available, otherwise use Qdrant ID
        pk_field = model_class._pk_field
        if pk_field and pk_field in data:
            # Original ID is already in the payload
            pass
        elif pk_field:
            # Use Qdrant ID as fallback
            data[pk_field] = point.id
        
        return model_class.from_dict(data)
    
    def get(self, model_class: Type[Base], id_value):
        """
        Get a model instance by ID
        
        Args:
            model_class: Model class to query
            id_value: ID value to look up
            
        Returns:
            Model instance or None if not found
        """
        collection = model_class.__collection__
        
        # Convert ID to Qdrant-compatible format
        qdrant_id = self._id_mapping.get(
            (collection, id_value), 
            _convert_id_for_qdrant(id_value)
        )
        
        result = self.client.retrieve(
            collection_name=collection,
            ids=[qdrant_id]
        )
        
        if not result:
            return None
        
        point = result[0]
        return self._point_to_model(point, model_class)
