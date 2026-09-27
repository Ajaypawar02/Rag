"""Streamlit chat UI: conversation history, session management, document upload, streaming answers.

Run:  streamlit run ui/streamlit_app.py     (API expected at $API_URL, default http://localhost:8080)
"""

import streamlit as st

from api_client import ApiError
from common import get_client, init_state, session_label, show_api_error, source_line, user_selector

st.set_page_config(page_title="Docs Assistant", page_icon="📚", layout="wide")
init_state()
api = get_client()


# --------------------------------------------------------------------------- #
# Sidebar: user, sessions, documents, memories
# --------------------------------------------------------------------------- #

def sidebar() -> None:
    st.sidebar.title("📚 Docs Assistant")
    user_id = user_selector()

    try:
        health = api.health()
        ok = health["status"] == "ok"
        st.sidebar.caption(("🟢 API healthy" if ok else "🟠 API degraded") +
                           f" · tracing {health['components'].get('langsmith_tracing', 'disabled')}")
    except ApiError as exc:
        st.sidebar.error(f"API unreachable: {exc}")
        st.stop()

    # --- Sessions ---
    st.sidebar.subheader("Sessions")
    if st.sidebar.button("➕ New session", width="stretch"):
        st.session_state.session_id = api.create_session(user_id)["session_id"]
        st.rerun()

    sessions = api.sessions(user_id)
    if sessions:
        ids = [s["session_id"] for s in sessions]
        if st.session_state.session_id not in ids:
            st.session_state.session_id = ids[0]
        labels = {s["session_id"]: session_label(s) for s in sessions}
        chosen = st.sidebar.radio("Your sessions", ids, index=ids.index(st.session_state.session_id),
                                  format_func=labels.get, label_visibility="collapsed")
        if chosen != st.session_state.session_id:
            st.session_state.session_id = chosen
            st.rerun()
        if st.sidebar.button("🗑️ Delete this session", width="stretch"):
            api.delete_session(user_id, st.session_state.session_id)
            st.session_state.session_id = None
            st.rerun()
    else:
        st.sidebar.caption("No sessions yet - ask a question to start one.")

    # --- Documents ---
    st.sidebar.subheader("Documents")
    uploaded = st.sidebar.file_uploader("Add documentation", type=["pdf", "md", "markdown", "html", "htm"])
    if uploaded is not None and st.sidebar.button("Upload & index", width="stretch"):
        with st.sidebar.status(f"Indexing {uploaded.name}…") as status:
            try:
                result = api.upload(uploaded.name, uploaded.getvalue())
                status.update(label=f"Indexed {result['source']}", state="complete")
                st.sidebar.success(
                    f"{result['chunks']} chunks ({result['chunks_with_code']} with code) · "
                    f"{result['doc_type']} · version {result['version'] or '-'} · {result['seconds']} s"
                )
            except ApiError as exc:
                status.update(label="Upload failed", state="error")
                show_api_error(exc)

    with st.sidebar.expander("Knowledge base"):
        for doc in api.documents():
            version = f" v{doc['version']}" if doc.get("version") else ""
            origin = " · uploaded" if doc.get("origin") == "upload" else ""
            st.caption(f"**{doc.get('title') or doc['source']}**{version} - {doc['chunks']} chunks{origin}")

    # --- Long-term memory ---
    with st.sidebar.expander("🧠 What I remember about you"):
        memories = api.memories(user_id)
        if not memories:
            st.caption("Nothing yet. Tell me about your stack or preferences.")
        for memory in memories:
            left, right = st.columns([5, 1])
            left.caption(f"{memory['text']}  \n*{memory.get('category') or ''}*")
            if right.button("✕", key=f"forget-{memory['key']}", help="Forget this"):
                api.forget(user_id, memory["key"])
                st.rerun()


# --------------------------------------------------------------------------- #
# Main: transcript + streaming chat
# --------------------------------------------------------------------------- #

def render_transcript(user_id: str, session_id: str) -> None:
    transcript = api.transcript(user_id, session_id)
    traces = {t["turn"]: t for t in api.decisions(user_id, session_id)}
    for turn in transcript:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"] or "_(no answer)_")
            if turn.get("sources"):
                st.caption("Sources: " + " · ".join(source_line(s) for s in turn["sources"]))
            trace = traces.get(turn["turn"])
            if trace:
                with st.expander(f"How this answer was produced · {trace['latency_ms'] / 1000:.1f} s"):
                    st.caption(f"intent **{trace['intent']}** · strategy **{trace['retrieval_strategy'] or 'none'}**"
                               + (f" · rewritten to *{trace['rewritten_query']}*" if trace.get("rewritten_query") else ""))
                    for d in trace["decisions"]:
                        st.markdown(f"- **{d['agent']}** ({d['latency_ms']:.0f} ms): {d['decision']}")


def stream_answer(user_id: str, session_id: str, prompt: str) -> None:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)
        errors: list[str] = []

        def tokens():
            answering = False
            for event, data in api.stream_chat(user_id, session_id, prompt):
                if event == "agent":
                    status.write(f"**{data['agent']}** · {data['decision']} ({data['latency_ms']:.0f} ms)")
                elif event == "token":
                    if not answering:
                        answering = True
                        status.update(label="Answering…", expanded=False)
                    yield data["text"]
                elif event == "error":
                    errors.append(data["detail"])

        try:
            st.write_stream(tokens())
        except ApiError as exc:
            errors.append(str(exc))
        if errors:
            status.update(label="Failed", state="error")
            st.error(errors[0])
            st.stop()
        status.update(label="Done", state="complete", expanded=False)
    st.rerun()  # re-render from the transcript: final answer with normalized citations and sources


def main() -> None:
    sidebar()
    user_id, session_id = st.session_state.user_id, st.session_state.session_id

    st.header("Ask the documentation")
    st.caption("Multi-turn RAG over FastAPI, Pydantic, Docker, Kubernetes, Redis, PostgreSQL, Python and LangGraph docs. "
               "Follow-up questions use the conversation; the assistant also remembers your preferences across sessions.")

    if session_id:
        try:
            render_transcript(user_id, session_id)
            memory = api.history(user_id, session_id)
            if memory["summary"]:
                with st.expander("🗜️ Conversation memory (summarized older turns)"):
                    st.caption(f"The model currently sees {len(memory['messages'])} recent messages plus this summary:")
                    st.markdown(memory["summary"])
        except ApiError as exc:
            show_api_error(exc)

    if prompt := st.chat_input("Ask about the docs…"):
        if not session_id:
            session_id = api.create_session(user_id)["session_id"]
            st.session_state.session_id = session_id
        stream_answer(user_id, session_id, prompt)


main()
