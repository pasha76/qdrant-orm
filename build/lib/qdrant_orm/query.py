from typing import Any, Dict, List, Optional, Tuple, Type, Union
from qdrant_orm import filters

from qdrant_client.http.models import Filter as QdrantFilter, MatchExcept, NamedVector, NamedSparseVector, SparseVector
from qdrant_client.http.models import SearchParams
from .base import Base, Field, VectorField
from .filters import Filter, FilterGroup
from qdrant_client.http.models import (
    Filter       as QdrantFilter,
    FieldCondition,
    Range,
    MatchValue,    # for exact match
    MatchAny,      # for "any" list‐match
    MatchExcept,   # for "not in" list‐match
    MatchText,     # for full-text search
    IsEmptyCondition,  # for checking empty fields
    IsNullCondition,   # for checking null fields
    HasIdCondition,    # for filtering by IDs
    ValuesCount,       # for filtering by array length
    NestedCondition,   # for nested object filtering
    Nested,            # for nested object filtering
)


class Query:
    """Query class for building and executing queries against Qdrant collections."""

    def __init__(self, session, model_class: Type[Base]):
        """Initialize a new Query instance.

        Args:
            session: The session to use for the query
            model_class: The model class to query
        """
        self._session = session
        self._model_class = model_class
        self._filters: List[Filter] = []
        self._vector_field: Optional[str] = None
        self._vector_value: Optional[List[float]] = None
        self._limit: int = 10
        self._offset: int = 0
        self._with_payload: bool = True
        self._with_vectors: bool = False
        self._score_threshold: Optional[float] = None

    def filter(self, *args: Filter) -> "Query":
        """Add filters to the query."""
        for arg in args:
            # Check if it's either our Filter type or the qdrant_orm Filter type
            if not (isinstance(arg, Filter) or isinstance(arg, qdrant_orm.filters.Filter)):
                raise TypeError(f"Expected Filter object, got {type(arg)}")
            self._filters.append(arg)
        return self

    def vector_search(
        self,
        field: Union[str, VectorField],
        vector: List[float] = None,
        query_vector: List[float] = None
    ) -> "Query":
        """Perform a vector search."""
        if isinstance(field, VectorField):
            self._vector_field = field.name
        else:
            self._vector_field = field

        if query_vector is not None:
            self._vector_value = query_vector
        elif vector is not None:
            self._vector_value = vector
        else:
            raise ValueError("Either 'vector' or 'query_vector' must be provided")
        return self

    def limit(self, limit: int) -> "Query":
        self._limit = limit
        return self

    def offset(self, offset: int) -> "Query":
        self._offset = offset
        return self

    def with_payload(self, with_payload: bool = True) -> "Query":
        self._with_payload = with_payload
        return self

    def with_vectors(self, with_vectors: bool = True) -> "Query":
        self._with_vectors = with_vectors
        return self

    def score_threshold(self, threshold: float) -> "Query":
        self._score_threshold = threshold
        return self

    def get(self, id_value: Any) -> Optional[Base]:
        client = self._session._get_client()
        collection_name = self._model_class.__collection__
        qdrant_id = self._session._convert_id_for_qdrant(id_value)
        try:
            result = client.retrieve(
                collection_name=collection_name,
                ids=[qdrant_id],
                with_payload=self._with_payload,
                with_vectors=self._with_vectors
            )
            if result and len(result) > 0:
                return self._session._point_to_model(result[0], self._model_class)
            return None
        except Exception as e:
            print(f"Error retrieving record: {e}")
            return None

    def all(self) -> List[Base]:
        client = self._session._get_client()
        collection_name = self._model_class.__collection__

        # 1) Combined search takes precedence
        if hasattr(self, "_combined_search_params"):
            return self._get_combined_search_results()

        # 2) Single-field vector search
        if self._vector_field and self._vector_value:
            if not isinstance(self._vector_value,dict):
                search_request = NamedVector(
                    name=self._vector_field,
                    vector=self._vector_value
                )
            else:
                print(self._vector_value)
                vec_dict = self._vector_value
                sparse_vec = SparseVector(
                    indices=vec_dict['indices'],
                    values=vec_dict['values']
                )
                search_request = NamedSparseVector(
                    name=self._vector_field.name,
                    vector=sparse_vec
                )
            search_params: Dict[str, Any] = {
                "collection_name": collection_name,
                "limit": self._limit,
                "offset": self._offset,
                "with_payload": self._with_payload,
                "with_vectors": self._with_vectors,
                "query_vector": search_request,
                "search_params":SearchParams(hnsw_ef=256)
            }

            if self._score_threshold is not None:
                search_params["score_threshold"] = self._score_threshold
            if self._build_qdrant_filter():
                search_params["query_filter"] = self._build_qdrant_filter() 

            try:
                results = client.search(**search_params)
                return [self._session._point_to_model(pt, self._model_class) for pt in results]
            except Exception as e:
                print(f"Error during vector search: {e}")
                return []

        # 3) Scroll for non-vector queries
        scroll_params: Dict[str, Any] = {
            "collection_name": collection_name,
            "limit": self._limit,
            "offset": self._offset,
            "with_payload": self._with_payload,
            "with_vectors": self._with_vectors,
        }
        if self._build_qdrant_filter():
            scroll_params["scroll_filter"] = self._build_qdrant_filter()
        try:
            points, _ = client.scroll(**scroll_params)
            return [self._session._point_to_model(pt, self._model_class) for pt in points]
        except Exception as e:
            print(f"Error during scroll: {e}")
            return []

    def first(self) -> Optional[Base]:
        results = self.limit(1).all()
        return results[0] if results else None

    def count(self) -> int:
        client = self._session._get_client()
        collection_name = self._model_class.__collection__
        count_params: Dict[str, Any] = {"collection_name": collection_name}
        if self._build_qdrant_filter():
            count_params["count_filter"] = self._build_qdrant_filter()
        try:
            result = client.count(**count_params)
            return result.count
        except Exception as e:
            print(f"Error counting records: {e}")
            return 0
    def _build_qdrant_filter(self) -> Optional[QdrantFilter]:
        if not self._filters:
            return None

        must, must_not, should = [], [], []

        for filt in self._filters:
            # handle groups
            if isinstance(filt, FilterGroup):
                for child in filt.filters:
                    cond = self._make_qdrant_condition(child)
                    if cond is None:  # Skip None conditions
                        continue
                    if isinstance(cond, list):  # Handle contains_all or not_in float conditions
                        if child.operator in ("not_in", "!="):
                            must_not.extend(cond)
                        else:
                            must.extend(cond)
                    else:
                        (must if filt.logic=="and" else should).append(cond)
                continue

            cond = self._make_qdrant_condition(filt)
            if cond is None:  # Skip None conditions
                continue
            
            if isinstance(cond, list):  # Handle contains_all or not_in float conditions
                if filt.operator in ("not_in", "!="):
                    must_not.extend(cond)
                else:
                    must.extend(cond)
            elif filt.operator in ("!=", "not_in"):
                must_not.append(cond)
            else:
                must.append(cond)

        # Always pass lists, never None
        return QdrantFilter(
            must=must,
            must_not=must_not,
            should=should
        )

    def _make_qdrant_condition(self, filt: Filter):
        key, op, val = filt.field_name, filt.operator, filt.value

        # Handle None values - skip the condition if value is None
        if val is None:
            return None

        if op == "==":
            return FieldCondition(key=key, match=MatchValue(value=val))

        if op == "!=" or op == "ne":
            return FieldCondition(key=key, match=MatchValue(value=val))

        if op == "in":
            # Ensure val is a list and use correct MatchAny syntax
            if not isinstance(val, (list, tuple)):
                val = [val]
            return FieldCondition(key=key, match=MatchAny(any=list(val)))

        if op == "not_in":
            # Ensure val is a list
            if not isinstance(val, (list, tuple)):
                val = [val]
            # Try to use model field type for robust casting
            field_type = None
            if hasattr(self, '_model_class') and hasattr(self._model_class, '_fields'):
                field_obj = self._model_class._fields.get(key)
                if field_obj and hasattr(field_obj, 'field_type'):
                    field_type = field_obj.field_type
            from qdrant_orm.types import Integer, Float, String, Boolean
            if field_type:
                if isinstance(field_type, Integer):
                    val = [int(v) for v in val]
                    return FieldCondition(key=key, match=MatchExcept(**{"except": val}))
                elif isinstance(field_type, String):
                    val = [str(v) for v in val]
                    return FieldCondition(key=key, match=MatchExcept(**{"except": val}))
                elif isinstance(field_type, Float):
                    # Float fields don't support MatchExcept in Qdrant
                    # and exact float matching is problematic due to precision
                    raise ValueError(
                        f"'not_in' filter is not supported for float field '{key}'. "
                        f"Qdrant does not support MatchExcept for float values. "
                        f"Consider using integer or string fields for exact matching, "
                        f"or use range filters (>, <, >=, <=) for float comparisons."
                    )
                elif isinstance(field_type, Boolean):
                    val = [bool(v) for v in val]
                    return FieldCondition(key=key, match=MatchExcept(**{"except": val}))
                else:
                    val = [str(v) for v in val]
                    return FieldCondition(key=key, match=MatchExcept(**{"except": val}))
            # Fallback: infer from first value
            if val:
                first = val[0]
                if isinstance(first, int):
                    val = [int(v) for v in val]
                elif isinstance(first, str):
                    val = [str(v) for v in val]
                elif isinstance(first, float):
                    # Float fields don't support MatchExcept in Qdrant
                    # and exact float matching is problematic due to precision
                    raise ValueError(
                        f"'not_in' filter is not supported for float field '{key}'. "
                        f"Qdrant does not support MatchExcept for float values. "
                        f"Consider using integer or string fields for exact matching, "
                        f"or use range filters (>, <, >=, <=) for float comparisons."
                    )
                else:
                    val = [str(v) for v in val]
            return FieldCondition(key=key, match=MatchExcept(**{"except": val}))

        if op == "contains":
            return FieldCondition(key=key, match=MatchValue(value=val))

        if op == "contains_any":
            # Convert values to strings for MatchAny
            if isinstance(val, (list, tuple)):
                val = [str(v) for v in val]
            return FieldCondition(key=key, match=MatchAny(any=val))

        if op == "contains_all":
            # For contains_all, we need to create multiple conditions with AND logic
            conditions = []
            if isinstance(val, (list, tuple)):
                for item in val:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=str(item))))
            return conditions

        if op in (">", ">=", "<", "<="):
            kwargs = {}
            if op == ">":   kwargs["gt"]  = val
            if op == ">=":  kwargs["gte"] = val
            if op == "<":   kwargs["lt"]  = val
            if op == "<=":  kwargs["lte"] = val
            return FieldCondition(key=key, range=Range(**kwargs))

        if op == "is_empty":
            from qdrant_client.http.models import PayloadField
            return IsEmptyCondition(is_empty=PayloadField(key=key))

        if op == "is_null":
            from qdrant_client.http.models import PayloadField
            return IsNullCondition(is_null=PayloadField(key=key))

        if op == "text_match":
            return FieldCondition(key=key, match=MatchText(text=val))

        if op == "values_count":
            # val should be a dict with gt, gte, lt, lte keys
            if isinstance(val, dict):
                return FieldCondition(key=key, values_count=ValuesCount(**val))
            else:
                raise ValueError(f"values_count operator requires a dict with gt/gte/lt/lte keys, got {type(val)}")

        raise ValueError(f"Unsupported operator: {op}")

    def _convert_filter_to_qdrant(self, filt: Filter) -> Dict[str,Any]:
        # no more FilterGroup handling here!
        field, op, value = filt.field_name, filt.operator, filt.value

        if op == "==":
            return {"key": field, "match": {"value": value}}
        if op in (">", ">=", "<", "<="):
            rng = {}
            if op == ">":   rng["gt"]  = value
            if op == ">=":  rng["gte"] = value
            if op == "<":   rng["lt"]  = value
            if op == "<=":  rng["lte"] = value
            return {"key": field, "range": rng}
        if op == "in":
            return {"key": field, "match": {"any": value}}
        if op == "not_in":
            return {"key": field, "match": {"except": value}}
        if op == "contains":
            return {"key": field, "match": {"value": value}}
        if op == "contains_any":
            return {"key": field, "match": {"any": value}}
        if op == "contains_all":
            return {"key": field, "match": {"all": value}}
        if op == "!=":
            return {"key": field, "match": {"value": value}}
        if op == "is_empty":
            return {"is_empty": {"key": field}}
        if op == "is_null":
            return {"is_null": {"key": field}}
        if op == "text_match":
            return {"key": field, "match": {"text": value}}
        if op == "values_count":
            return {"key": field, "values_count": value}

        raise ValueError(f"Unsupported operator: {op}")

    def combined_vector_search(
        self,
        vector_fields_with_weights: Dict[Union[str, VectorField], float],
        query_vectors: Dict[str, List[float]],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> "Query":
        """Perform a combined vector search across multiple vector fields with weights."""
        self._combined_search_params = {
            "vector_fields_with_weights": vector_fields_with_weights,
            "query_vectors": query_vectors,
            "limit": limit,
            "score_threshold": score_threshold,
        }
        return self

    def _execute_combined_vector_search(self) -> List[Tuple[Any, float]]:
        params = self._combined_search_params
        client = self._session._get_client()
        collection_name = self._model_class.__collection__
        # Normalize weights
        weights = {
            (f.name if isinstance(f, VectorField) else f): w
            for f, w in params["vector_fields_with_weights"].items()
        }
        total = sum(weights.values())
        normalized = {f: w/total for f, w in weights.items()}
        all_scores: Dict[Any, float] = {}
        for fname, weight in normalized.items():
            if weight <= 0 or fname not in params["query_vectors"]:
                continue
            qv = params["query_vectors"][fname]
            nv = NamedVector(name=fname, vector=qv)
            sp: Dict[str, Any] = {
                "collection_name": collection_name,
                "limit": params["limit"] * 3,
                "with_payload": False,
                "with_vectors": False,
                "query_vector": nv,
            }
            if params["score_threshold"] is not None:
                sp["score_threshold"] = params["score_threshold"]
            if self._build_qdrant_filter():
                sp["query_filter"] = self._build_qdrant_filter()
            try:
                res = client.search(**sp)
                for pt in res:
                    pid = pt.id
                    all_scores[pid] = all_scores.get(pid, 0.0) + pt.score * weight
            except Exception as e:
                print(f"Error during vector search for {fname}: {e}")
        # Sort & limit
        sorted_pts = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pts[: params["limit"]]

    def _get_combined_search_results(self) -> List[Base]:
        combined = self._execute_combined_vector_search()
        if not combined:
            return []
        client = self._session._get_client()
        collection_name = self._model_class.__collection__
        try:
            ids = [pid for pid, _ in combined]
            points = client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=self._with_payload,
                with_vectors=self._with_vectors,
            )
            id_map = {str(pt.id): pt for pt in points}
            ordered = [id_map.get(str(pid)) for pid, _ in combined]
            return [self._session._point_to_model(pt, self._model_class) for pt in ordered if pt]
        except Exception as e:
            print(f"Error retrieving combined search results: {e}")
            return []

