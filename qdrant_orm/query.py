from typing import Any, Dict, List, Optional, Tuple, Type, Union

from qdrant_client.http.models import Filter as QdrantFilter, NamedVector, NamedSparseVector, SparseVector

from .base import Base, Field, VectorField
from .filters import Filter, FilterGroup


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
            if not isinstance(arg, Filter):
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
            }

            if self._score_threshold is not None:
                search_params["score_threshold"] = self._score_threshold
            if self._build_qdrant_filter():
                search_params["filter"] = self._build_qdrant_filter()

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
        must_conditions: List[Dict[str, Any]] = []
        for filt in self._filters:
            qf = self._convert_filter_to_qdrant(filt)
            if qf:
                must_conditions.append(qf)
        return QdrantFilter(must=must_conditions) if must_conditions else None

    def _convert_filter_to_qdrant(self, filter_obj: Filter) -> Dict[str, Any]:
        if isinstance(filter_obj, FilterGroup):
            if filter_obj.logic == "and":
                return {"must": [self._convert_filter_to_qdrant(f) for f in filter_obj.filters]}
            if filter_obj.logic == "or":
                return {"should": [self._convert_filter_to_qdrant(f) for f in filter_obj.filters]}
            raise ValueError(f"Unsupported filter logic: {filter_obj.logic}")
        field_name, op, value = filter_obj.field_name, filter_obj.operator, filter_obj.value
        if op == "==":
            return {"key": field_name, "match": {"value": value}}
        if op == "!=":
            return {"must_not": {"key": field_name, "match": {"value": value}}}
        if op in (">", ">=", "<", "<="):
            range_cond = {}
            if op == ">":   range_cond["gt"] = value
            if op == ">=":  range_cond["gte"] = value
            if op == "<":   range_cond["lt"] = value
            if op == "<=":  range_cond["lte"] = value
            return {"key": field_name, "range": range_cond}
        if op == "in":
            return {"key": field_name, "match": {"any": value}}
        if op == "not_in":
            return {"must_not": {"key": field_name, "match": {"any": value}}}
        if op in ("contains", "contains_all", "contains_any"):
            mode = "value" if op == "contains" else ("all" if op == "contains_all" else "any")
            return {"key": field_name, "match": {mode: value}}
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
