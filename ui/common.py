"""Shared Streamlit helpers: API client, user identity and session selection across pages."""

import streamlit as st

from api_client import ApiClient, ApiError


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def init_state() -> None:
    st.session_state.setdefault("user_id", "demo")
    st.session_state.setdefault("session_id", None)


def user_selector() -> str:
    """Sidebar user id input; switching user clears the selected session."""
    previous = st.session_state.user_id
    user_id = st.sidebar.text_input("User ID", value=previous, help="Sessions and memories are isolated per user.")
    user_id = user_id.strip() or "demo"
    if user_id != previous:
        st.session_state.user_id = user_id
        st.session_state.session_id = None
    return user_id


def session_label(session: dict) -> str:
    title = session.get("title") or "New conversation"
    return f"{title[:40]}{'…' if len(title) > 40 else ''}  ·  {session['turns']} turns"


def show_api_error(exc: ApiError) -> None:
    st.error(f"API error - {exc}")


def source_line(source: dict) -> str:
    where = source.get("section") or (f"page {source['page']}" if source.get("page") else "")
    return f"[{source['index']}] `{source.get('source')}`" + (f" - {where}" if where else "")
