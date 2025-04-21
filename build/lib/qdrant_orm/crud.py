"""
CRUD operations for Qdrant ORM
"""
from typing import Dict, Any, Type, List, Optional, Union, Tuple, Callable
import uuid

from qdrant_client.http import models as qmodels

from .base import Base


class CRUDOperations:
    """Helper class for CRUD operations"""
    
    @staticmethod
    def bulk_insert(session, instances: List[Base], batch_size: int = 100):
        """
        Insert multiple instances in batches
        
        Args:
            session: QdrantSession instance
            instances: List of model instances to insert
            batch_size: Number of instances to insert in each batch
        """
        # Group instances by model class
        instances_by_class = {}
        for instance in instances:
            class_name = instance.__class__.__name__
            if class_name not in instances_by_class:
                instances_by_class[class_name] = []
            instances_by_class[class_name].append(instance)
        
        # Process each model class separately
        for instances in instances_by_class.values():
            # Process in batches
            for i in range(0, len(instances), batch_size):
                batch = instances[i:i+batch_size]
                for instance in batch:
                    session.add(instance)
                session.commit()
    
    @staticmethod
    def bulk_update(session, instances: List[Base], batch_size: int = 100):
        """
        Update multiple instances in batches
        
        Args:
            session: QdrantSession instance
            instances: List of model instances to update
            batch_size: Number of instances to update in each batch
        """
        # Same implementation as bulk_insert since upsert is used
        CRUDOperations.bulk_insert(session, instances, batch_size)
    
    @staticmethod
    def bulk_delete(session, instances: List[Base], batch_size: int = 100):
        """
        Delete multiple instances in batches
        
        Args:
            session: QdrantSession instance
            instances: List of model instances to delete
            batch_size: Number of instances to delete in each batch
        """
        # Group instances by model class
        instances_by_class = {}
        for instance in instances:
            class_name = instance.__class__.__name__
            if class_name not in instances_by_class:
                instances_by_class[class_name] = []
            instances_by_class[class_name].append(instance)
        
        # Process each model class separately
        for instances in instances_by_class.values():
            # Process in batches
            for i in range(0, len(instances), batch_size):
                batch = instances[i:i+batch_size]
                for instance in batch:
                    session.delete(instance)
                session.commit()
    
    @staticmethod
    def delete_by_filter(session, model_class: Type[Base], *filters):
        """
        Delete instances matching the given filters
        
        Args:
            session: QdrantSession instance
            model_class: Model class to delete from
            *filters: Filter conditions
        """
        # First query to get all matching instances
        query = session.query(model_class).filter(*filters)
        instances = query.all()
        
        # Then delete them
        for instance in instances:
            session.delete(instance)
        session.commit()
    
    @staticmethod
    def update_by_filter(session, model_class: Type[Base], update_data: Dict[str, Any], *filters):
        """
        Update instances matching the given filters
        
        Args:
            session: QdrantSession instance
            model_class: Model class to update
            update_data: Dictionary of field names and values to update
            *filters: Filter conditions
        """
        # First query to get all matching instances
        query = session.query(model_class).filter(*filters)
        instances = query.all()
        
        # Then update them
        for instance in instances:
            for field_name, value in update_data.items():
                setattr(instance, field_name, value)
            session.add(instance)
        session.commit()
    
    @staticmethod
    def get_or_create(session, model_class: Type[Base], defaults: Dict[str, Any] = None, **kwargs):
        """
        Get an instance matching the given filters, or create one if it doesn't exist
        
        Args:
            session: QdrantSession instance
            model_class: Model class to query
            defaults: Dictionary of field names and values to use when creating
            **kwargs: Filter conditions
            
        Returns:
            Tuple of (instance, created) where created is a boolean indicating whether the instance was created
        """
        defaults = defaults or {}
        
        # Build filters from kwargs
        filters = []
        from .query import Filter
        for field_name, value in kwargs.items():
            filters.append(Filter(field_name, "==", value))
        
        # Try to get existing instance
        query = session.query(model_class).filter(*filters)
        instance = query.first()
        
        if instance:
            return instance, False
        
        # Create new instance
        create_data = dict(kwargs)
        create_data.update(defaults)
        instance = model_class(**create_data)
        session.add(instance)
        session.commit()
        
        return instance, True
    
    @staticmethod
    def update_or_create(session, model_class: Type[Base], defaults: Dict[str, Any] = None, **kwargs):
        """
        Update an instance matching the given filters, or create one if it doesn't exist
        
        Args:
            session: QdrantSession instance
            model_class: Model class to query
            defaults: Dictionary of field names and values to use when updating or creating
            **kwargs: Filter conditions
            
        Returns:
            Tuple of (instance, created) where created is a boolean indicating whether the instance was created
        """
        defaults = defaults or {}
        
        # Build filters from kwargs
        filters = []
        from .query import Filter
        for field_name, value in kwargs.items():
            filters.append(Filter(field_name, "==", value))
        
        # Try to get existing instance
        query = session.query(model_class).filter(*filters)
        instance = query.first()
        
        if instance:
            # Update instance
            for field_name, value in defaults.items():
                setattr(instance, field_name, value)
            session.add(instance)
            session.commit()
            return instance, False
        
        # Create new instance
        create_data = dict(kwargs)
        create_data.update(defaults)
        instance = model_class(**create_data)
        session.add(instance)
        session.commit()
        
        return instance, True
