"""Schema and tabular field normalization helpers."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

class SchemaMapper:
    @classmethod
    def flatten_payload(cls, payload: Any, prefix: str = "") -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        flattened: Dict[str, Any] = {}
        for key, value in payload.items():
            raw_key = str(key or "").strip()
            if not raw_key:
                continue
            next_prefix = f"{prefix}_{raw_key}" if prefix else raw_key
            if isinstance(value, dict):
                flattened.update(cls.flatten_payload(value, next_prefix))
                continue
            if isinstance(value, list) and value and all(not isinstance(item, (dict, list)) for item in value):
                flattened[next_prefix] = ", ".join(str(item) for item in value if str(item).strip())
                continue
            flattened[next_prefix] = value
        return flattened

    @staticmethod
    def normalize_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    @classmethod
    def _token_set(cls, value: Any) -> Set[str]:
        return set(filter(None, cls.normalize_key(value).split("_")))

    @classmethod
    def _resolve_alias(cls, raw_key: Any, alias_map: Dict[str, List[str]]) -> Optional[str]:
        normalized_key = cls.normalize_key(raw_key)
        if not normalized_key:
            return None

        direct_hits: List[str] = []
        fuzzy_hits: List[Tuple[int, str]] = []
        raw_tokens = cls._token_set(raw_key)

        for canonical, aliases in (alias_map or {}).items():
            for alias in aliases or []:
                alias_key = cls.normalize_key(alias)
                if not alias_key:
                    continue
                if normalized_key == alias_key:
                    direct_hits.append(canonical)
                    break
                alias_tokens = cls._token_set(alias)
                if raw_tokens and alias_tokens and (alias_tokens <= raw_tokens or raw_tokens <= alias_tokens):
                    fuzzy_hits.append((len(alias_tokens & raw_tokens), canonical))

        if direct_hits:
            return direct_hits[0]
        if fuzzy_hits:
            fuzzy_hits.sort(reverse=True)
            return fuzzy_hits[0][1]
        return None

    @classmethod
    def normalize_record(
        cls,
        payload: Optional[Dict[str, Any]],
        alias_map: Dict[str, List[str]],
        passthrough_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        raw = dict(payload or {})
        flattened = cls.flatten_payload(raw)
        raw.update({k: v for k, v in flattened.items() if k not in raw})
        normalized: Dict[str, Any] = {}
        passthrough = passthrough_keys or []

        for key, value in raw.items():
            mapped_key = cls._resolve_alias(key, alias_map)
            target_key = mapped_key or cls.normalize_key(key)
            if mapped_key and normalized.get(mapped_key) not in (None, "", [], {}):
                continue
            normalized[target_key] = value

        for key in passthrough:
            if key in raw and key not in normalized:
                normalized[key] = raw[key]
        return normalized

    @classmethod
    def remap_dataframe(cls, df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> pd.DataFrame:
        renamed = df.copy()
        rename_map: Dict[str, str] = {}
        taken_targets = set(renamed.columns)

        for column in renamed.columns:
            target = cls._resolve_alias(column, alias_map)
            if not target or column == target or target in taken_targets:
                continue
            rename_map[column] = target
            taken_targets.add(target)

        if rename_map:
            renamed = renamed.rename(columns=rename_map)
        return renamed


