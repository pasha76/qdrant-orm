"""
Engine and session classes for Qdrant ORM
"""
from typing import Dict, Any, Type, List, Optional, Union, Tuple
import uuid
import re

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import SparseVector
from .base import Base,VectorField,SparseVectorField


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
        Create (or recreate) a Qdrant collection wiring up ALL vector fields
        by their attribute names—dense and sparse—so that your upserts
        with named vectors will always succeed.
        """
        # 1) Gather your fields
        dense_fields = {
            name: fld
            for name, fld in model_class._fields.items()
            if isinstance(fld, VectorField)
        }
        sparse_fields = {
            name: fld
            for name, fld in model_class._fields.items()
            if isinstance(fld, SparseVectorField)
        }

        # 2) Build a **named** vectors_config for every dense field
        #    (never use the single-vector shorthand)
        vectors_config = {
            name: qmodels.VectorParams(size=fld.dimensions, distance=fld.distance)
            for name, fld in dense_fields.items()
        }

        # 3) Build named sparse_vectors_config for every sparse field
        sparse_vectors_config = {
            name: qmodels.SparseVectorParams()
            for name in sparse_fields
        }

        # 4) Recreate the collection with both named configs
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
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
            operations_by_collection.setdefault(collection, {'add': [], 'delete': []})[op].append(instance)

        for collection, operations in operations_by_collection.items():
            # Process additions
            if operations['add']:
                points = []
                for instance in operations['add']:
                    vectors = {}
                    payload = {}

                    # Split out both dense and sparse vector fields
                    for name, value in instance._values.items():
                        field = instance.__class__._fields.get(name)
                        if isinstance(field, (VectorField, SparseVectorField)):
                            vectors[name] = value
                        else:
                            payload[name] = value

                    # Ensure primary key
                    original_id = getattr(instance, instance.__class__._pk_field, None)
                    if original_id is None:
                        original_id = str(uuid.uuid4())
                        setattr(instance, instance.__class__._pk_field, original_id)
                    qdrant_id = _convert_id_for_qdrant(original_id)
                    self._id_mapping[(collection, original_id)] = qdrant_id
                    payload[instance.__class__._pk_field] = original_id

                    # Build PointStruct with named vectors (dense or sparse)
                    if len(vectors) == 1 and isinstance(next(iter(instance.__class__._fields.values())), VectorField):
                        # single dense vector can be flattened
                        vector_value = next(iter(vectors.values()))
                    else:
                        # multiple or sparse vectors remain named
                        vector_value = vectors

                    points.append(qmodels.PointStruct(
                        id=qdrant_id,
                        vector=vector_value,
                        payload=payload
                    ))

                self.client.upsert(collection_name=collection, points=points)

            # Process deletions
            if operations['delete']:
                ids = []
                for instance in operations['delete']:
                    orig = instance.pk
                    q_id = self._id_mapping.get((collection, orig), _convert_id_for_qdrant(orig))
                    ids.append(q_id)
                self.client.delete(
                    collection_name=collection,
                    points_selector=qmodels.PointIdsList(points=ids)
                )

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
                    if isinstance(vector, SparseVector):
                        data[name] = vector.model_dump()
                    else:
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
        
        # Include score if available (from search results)
        if hasattr(point, 'score') and point.score is not None:
            data['score'] = point.score
        
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
