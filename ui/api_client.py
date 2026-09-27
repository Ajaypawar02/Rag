"""Thin HTTP client for the FastAPI backend, shared by all Streamlit pages.

The UI never imports the `app` package: it talks to the API only, exactly
like any other client, so it can run in its own container.
"""

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8080")


class ApiError(Exception):
    pass


class ApiClient:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(180, connect=5))

    # --- plumbing -------------------------------------------------------------

    def _request(self, method: str, path: str, user_id: str | None = None, **kwargs) -> Any:
        headers = {"X-User-Id": user_id} if user_id else {}
        try:
            response = self._http.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{response.status_code}: {detail}")
        return response.json() if response.content else None

    # --- system ---------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def metrics(self) -> dict:
        return self._request("GET", "/metrics")

    # --- documents --------------------------------------------------------------

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload(self, filename: str, content: bytes) -> dict:
        return self._request("POST", "/documents/upload", files={"file": (filename, content)})

    # --- sessions ---------------------------------------------------------------

    def create_session(self, user_id: str, title: str | None = None) -> dict:
        return self._request("POST", "/sessions", user_id, json={"title": title})

    def sessions(self, user_id: str) -> list[dict]:
        return self._request("GET", "/sessions", user_id)

    def delete_session(self, user_id: str, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}", user_id)

    def transcript(self, user_id: str, session_id: str) -> list[dict]:
        return self._request("GET", f"/sessions/{session_id}/transcript", user_id)

    def history(self, user_id: str, session_id: str) -> dict:
        return self._request("GET", f"/sessions/{session_id}/history", user_id)

    def decisions(self, user_id: str, session_id: str) -> list[dict]:
        return self._request("GET", f"/sessions/{session_id}/decisions", user_id)

    # --- memory -----------------------------------------------------------------

    def memories(self, user_id: str) -> list[dict]:
        return self._request("GET", "/memories", user_id)

    def forget(self, user_id: str, key: str) -> None:
        self._request("DELETE", f"/memories/{key}", user_id)

    # --- chat -------------------------------------------------------------------

    def stream_chat(self, user_id: str, session_id: str, message: str) -> Iterator[tuple[str, dict]]:
        """Yield (event, data) pairs from the SSE endpoint: agent / token / done / error."""
        try:
            with self._http.stream(
                "POST", f"/sessions/{session_id}/chat/stream",
                headers={"X-User-Id": user_id}, json={"message": message},
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(f"{response.status_code}: {response.json().get('detail', response.text)}")
                event = "message"
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        event = line[len("event: "):]
                    elif line.startswith("data: "):
                        yield event, json.loads(line[len("data: "):])
        except httpx.HTTPError as exc:
            raise ApiError(f"Connection to the API failed: {exc}") from exc
