"""
Base classes for Qdrant ORM
"""
from typing import Dict, Any, Type, ClassVar, Optional, List, Set, get_type_hints
import inspect

from .filters import Filter


class MetaData:
    """Container for schema information"""
    
    def __init__(self):
        self.collections = {}
    
    def create_all(self, engine):
        """Create all collections defined in the metadata"""
        for collection_name, model_class in self.collections.items():
            engine.create_collection(collection_name, model_class)
    
    def drop_all(self, engine):
        """Drop all collections defined in the metadata"""
        for collection_name in self.collections:
            engine.drop_collection(collection_name)


class Field:
    """Descriptor for model fields"""
    
    def __init__(self, field_type, primary_key=False, nullable=True, default=None):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.name = None
        self.owner = None
    
    def __set_name__(self, owner, name):
        self.name = name
        self.owner = owner
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._values.get(self.name, self.default)
    
    def __set__(self, instance, value):
        if value is None and not self.nullable:
            raise ValueError(f"Field '{self.name}' cannot be None")
        
        # Type checking could be added here
        instance._values[self.name] = value
    
    # Operator overloading for filtering
    def __eq__(self, other):
        return Filter(self.name, "==", other)
    
    def __ne__(self, other):
        return Filter(self.name, "!=", other)
    
    def __gt__(self, other):
        return Filter(self.name, ">", other)
    
    def __ge__(self, other):
        return Filter(self.name, ">=", other)
    
    def __lt__(self, other):
        return Filter(self.name, "<", other)
    
    def __le__(self, other):
        return Filter(self.name, "<=", other)
    
    def in_(self, values):
        """Check if field value is in a list of values"""
        return Filter(self.name, "in", values)
    
    def not_in(self, values):
        """Check if field value is not in a list of values"""
        return Filter(self.name, "not_in", values)


class ArrayField(Field):
    """Field for array data"""
    
    def __init__(self, field_type, nullable=True, default=None):
        """
        Initialize an array field
        
        Args:
            field_type: The data type of elements in the array
            nullable: Whether the array itself can be null
            default: Default value (defaults to empty list if None)
        """
        # Default for arrays should be an empty list if not specified
        if default is None:
            default = []
            
        super().__init__(field_type=field_type, primary_key=False, nullable=nullable, default=default)
    
    def __set__(self, instance, value):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Array field '{self.name}' must be a list")
            
            # Validate each element in the array if we have a field_type with validation
            if hasattr(self.field_type, 'validate'):
                for item in value:
                    if not self.field_type.validate(item):
                        raise ValueError(f"Invalid value in array field '{self.name}': {item}")
        
        super().__set__(instance, value)
    
    # Override equality operators for array fields to use contains operator
    def __eq__(self, other):
        """For array fields, equality means contains the value"""
        return Filter(self.name, "contains", other)
    
    def contains(self, value):
        """Check if array contains a value"""
        return Filter(self.name, "contains", value)
    
    def contains_all(self, values):
        """Check if array contains all specified values"""
        return Filter(self.name, "contains_all", values)
    
    def contains_any(self, values):
        """Check if array contains any of the specified values"""
        return Filter(self.name, "contains_any", values)


class VectorField(Field):
    """Special field for vector data"""
    
    def __init__(self, dimensions, distance="Cosine", **kwargs):
        super().__init__(field_type="vector", **kwargs)
        self.dimensions = dimensions
        self.distance = distance
    
    def __set__(self, instance, value):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Vector field '{self.name}' must be a list")
            if len(value) != self.dimensions:
                raise ValueError(f"Vector field '{self.name}' must have exactly {self.dimensions} dimensions")
        super().__set__(instance, value)
    def __eq__(self, other):
        # Two VectorFields are equal if they have the same name, dims, and distance metric
        return (
            isinstance(other, VectorField) and
            self.name == other.name and
            self.dimensions == other.dimensions and
            self.distance == other.distance
        )

    def __hash__(self):
        # Hash on a tuple of class, name, dims, and distance
        return hash((self.__class__, self.name, self.dimensions, self.distance))

    def __str__(self):
        return f"{self.name}:{self.dimensions}:{self.distance}"
    
    def __call__(self, *args, **kwargs):
        # Return the field name as a plain string
        print("--------------------------------")
        print(self.name)
        print("--------------------------------")
        return self.name

class ModelMeta(type):
    """Metaclass for ORM models"""
    
    def __new__(mcs, name, bases, attrs):
        if name == "Base" or name.startswith("_"):
            return super().__new__(mcs, name, bases, attrs)
        
        # Create new class
        cls = super().__new__(mcs, name, bases, attrs)
        
        # Set collection name if not explicitly defined
        if not hasattr(cls, "__collection__"):
            cls.__collection__ = name.lower()
        
        # Register model with metadata
        if hasattr(cls, "metadata") and cls.metadata is not None:
            cls.metadata.collections[cls.__collection__] = cls
        
        # Collect fields
        cls._fields = {}
        cls._pk_field = None
        
        for key, value in attrs.items():
            if isinstance(value, Field):
                cls._fields[key] = value
                if value.primary_key:
                    if cls._pk_field is not None:
                        raise ValueError(f"Multiple primary keys defined for {name}")
                    cls._pk_field = key
        
        # Inherit fields from parent classes
        for base in bases:
            if hasattr(base, "_fields"):
                for key, value in base._fields.items():
                    if key not in cls._fields:
                        cls._fields[key] = value
                        if value.primary_key and cls._pk_field is None:
                            cls._pk_field = key
        
        return cls


class Base(metaclass=ModelMeta):
    """Base class for all models"""
    
    metadata: ClassVar[MetaData] = MetaData()
    
    def __init__(self, **kwargs):
        self._values = {}
        
        # Set default values
        for name, field in self.__class__._fields.items():
            if field.default is not None:
                self._values[name] = field.default
        
        # Set provided values
        for name, value in kwargs.items():
            if name in self.__class__._fields:
                setattr(self, name, value)
            else:
                raise AttributeError(f"Unknown field '{name}' for {self.__class__.__name__}")
    
    def __repr__(self):
        values = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"{self.__class__.__name__}({values})"
    
    @property
    def pk(self):
        """Get the primary key value"""
        if self.__class__._pk_field is None:
            raise ValueError(f"No primary key defined for {self.__class__.__name__}")
        return getattr(self, self.__class__._pk_field)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return dict(self._values)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Base':
        """Create model instance from dictionary"""
        return cls(**data)
