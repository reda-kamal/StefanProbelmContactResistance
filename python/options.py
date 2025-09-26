"""Helper for nested option structs implemented as dictionaries."""
from __future__ import annotations
from typing import Any, Dict


def get_opt(opts: Dict[str, Any], field: str, default: Any) -> Any:
    if isinstance(opts, dict) and field in opts and opts[field] is not None:
        return opts[field]
    return default


def get_struct(opts: Dict[str, Any], field: str) -> Dict[str, Any]:
    if isinstance(opts, dict) and field in opts and isinstance(opts[field], dict):
        return opts[field]
    return {}
