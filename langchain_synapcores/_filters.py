"""Translate LangChain-style metadata filter dicts into SynapCores SQL."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*(\.[A-Za-z_][A-Za-z0-9_\-]*)*$")

_COMPARISONS = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def _path(key: str, column: str) -> str:
    if not _KEY.match(key):
        raise ValueError(f"Invalid metadata key {key!r} in filter.")
    return f"json_extract({column}, '$.{key}')"


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"Filter values must be str, int, float or bool, got {type(value).__name__}.")


def _field(key: str, cond: Any, column: str, params: List[Any]) -> str:
    expr = _path(key, column)
    if not isinstance(cond, dict):
        cond = {"$eq": cond}
    parts: List[str] = []
    for op, value in cond.items():
        if op in _COMPARISONS:
            params.append(_scalar(value))
            parts.append(f"{expr} {_COMPARISONS[op]} ${len(params)}")
        elif op in ("$in", "$nin"):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{op} needs a non-empty list.")
            holders = []
            for v in value:
                params.append(_scalar(v))
                holders.append(f"${len(params)}")
            neg = "NOT " if op == "$nin" else ""
            parts.append(f"{expr} {neg}IN ({', '.join(holders)})")
        else:
            raise ValueError(f"Unknown filter operator {op!r}.")
    return parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")"


def build_where(
    filter: Dict[str, Any] | None, column: str = "metadata", params: List[Any] | None = None
) -> Tuple[str, List[Any]]:
    """Return ``(sql, params)``. ``sql`` is an expression, or ``""`` when there is no filter.

    Supports field equality (``{"k": "v"}``), comparison operators
    (``$eq $ne $gt $gte $lt $lte $in $nin``) and the logical operators ``$and`` / ``$or``.
    Parameters are numbered after any existing entries in ``params``.
    """
    params = params if params is not None else []
    if not filter:
        return "", params

    def walk(f: Dict[str, Any]) -> str:
        clauses: List[str] = []
        for key, value in f.items():
            if key in ("$and", "$or"):
                if not isinstance(value, list) or not value:
                    raise ValueError(f"{key} needs a non-empty list of filters.")
                subs = [walk(v) for v in value]
                joiner = " AND " if key == "$and" else " OR "
                clauses.append("(" + joiner.join(subs) + ")")
            elif key.startswith("$"):
                raise ValueError(f"Unknown logical operator {key!r}.")
            else:
                clauses.append(_field(key, value, column, params))
        return clauses[0] if len(clauses) == 1 else "(" + " AND ".join(clauses) + ")"

    return walk(filter), params
