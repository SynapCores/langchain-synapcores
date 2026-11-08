"""Small HTTP client for the SynapCores gateway REST API."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import httpx

DEFAULT_URL = "http://127.0.0.1:8080"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class SynapCoresError(RuntimeError):
    """Error returned by the SynapCores gateway."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


def safe_identifier(name: str, kind: str = "identifier") -> str:
    """Validate a SQL identifier (table or database name)."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(
            f"Invalid {kind} {name!r}: use letters, digits and underscores, starting with a letter or underscore."
        )
    return name


def vector_literal(vector: Sequence[float]) -> str:
    """Render a float vector as a SynapCores SQL vector literal."""
    parts = []
    for v in vector:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("Embedding values must be finite numbers.")
        parts.append(repr(f))
    return "[" + ", ".join(parts) + "]"


class SynapCoresClient:
    """Synchronous client for ``/v1/query/execute`` and ``/v1/ai/embeddings``.

    Authentication, in order of preference:

    * ``api_key`` (or env ``SYNAPCORES_API_KEY``) or ``token`` (env ``SYNAPCORES_TOKEN``),
      sent as a Bearer token.
    * ``username`` and ``password`` (env ``SYNAPCORES_USERNAME`` / ``SYNAPCORES_PASSWORD``):
      the client logs in and renews the session automatically.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 60.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.url = (url or os.environ.get("SYNAPCORES_URL") or DEFAULT_URL).rstrip("/")
        self._token = (
            api_key
            or token
            or os.environ.get("SYNAPCORES_API_KEY")
            or os.environ.get("SYNAPCORES_TOKEN")
        )
        self._username = username or os.environ.get("SYNAPCORES_USERNAME")
        self._password = password or os.environ.get("SYNAPCORES_PASSWORD")
        self._http = http_client or httpx.Client(base_url=self.url, timeout=timeout)

    # -- auth ---------------------------------------------------------------

    def _can_login(self) -> bool:
        return bool(self._username and self._password)

    def _login(self) -> None:
        if not self._can_login():
            raise SynapCoresError(
                "No SynapCores credentials: pass api_key, or username and password "
                "(or set SYNAPCORES_API_KEY / SYNAPCORES_USERNAME and SYNAPCORES_PASSWORD)."
            )
        r = self._http.post(
            "/v1/auth/login", json={"username": self._username, "password": self._password}
        )
        if r.status_code != 200:
            raise SynapCoresError(f"Login failed (HTTP {r.status_code})", status=r.status_code)
        self._token = r.json().get("access_token")
        if not self._token:
            raise SynapCoresError("Login response had no access_token")

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        if not self._token:
            self._login()
        r = self._http.post(path, json=payload, headers={"Authorization": f"Bearer {self._token}"})
        if r.status_code == 401 and self._can_login():
            self._login()
            r = self._http.post(
                path, json=payload, headers={"Authorization": f"Bearer {self._token}"}
            )
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code >= 400:
            err = (body or {}).get("error") if isinstance(body, dict) else None
            if isinstance(err, dict):
                raise SynapCoresError(
                    err.get("message") or f"HTTP {r.status_code}",
                    status=r.status_code,
                    code=err.get("code"),
                )
            raise SynapCoresError(f"HTTP {r.status_code}: {r.text[:300]}", status=r.status_code)
        if isinstance(body, dict) and "data" in body and isinstance(body["data"], (dict, list)):
            return body["data"]
        return body

    # -- API ----------------------------------------------------------------

    def execute(
        self,
        sql: str,
        parameters: Optional[Sequence[Any]] = None,
        *,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run one SQL statement and return rows as dicts."""
        payload: Dict[str, Any] = {"sql": sql, "parameters": list(parameters or [])}
        if database:
            payload["database"] = database
        data = self._post("/v1/query/execute", payload) or {}
        cols = [c.get("name") if isinstance(c, dict) else str(c) for c in data.get("columns") or []]
        rows = data.get("rows") or []
        out: List[Dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(row)
            else:
                out.append({cols[i] if i < len(cols) else str(i): v for i, v in enumerate(row)})
        return out

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Embed texts with the engine's embedding model."""
        if not texts:
            return []
        payload: Dict[str, Any] = {"texts": texts}
        if model:
            payload["model"] = model
        data = self._post("/v1/ai/embeddings/batch", payload)
        embeddings = data.get("embeddings") if isinstance(data, dict) else data
        return [list(map(float, e)) for e in embeddings]

    def close(self) -> None:
        self._http.close()
