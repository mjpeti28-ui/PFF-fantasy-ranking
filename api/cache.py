from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable, Tuple


class QueryCache:
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[Hashable, Any] = OrderedDict()

    def _prune(self) -> None:
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def get(self, key: Hashable) -> Any | None:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        self._prune()


query_cache = QueryCache()


def build_query_key(
    namespace: str,
    *,
    context_timestamp: str,
    filters: Tuple[str, ...],
    sort: str | None,
    columns: Tuple[str, ...],
    limit: int,
    offset: int,
) -> Tuple[Any, ...]:
    return (
        namespace,
        context_timestamp,
        filters,
        sort,
        columns,
        int(limit),
        int(offset),
    )
