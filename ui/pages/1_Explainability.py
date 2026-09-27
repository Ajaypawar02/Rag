"""Explainability dashboard: what every agent decided, why, and how long it took - per turn."""

import pandas as pd
import streamlit as st

from api_client import ApiError
from common import get_client, init_state, session_label, show_api_error, user_selector

st.set_page_config(page_title="Explainability", page_icon="🔎", layout="wide")
init_state()
api = get_client()

st.sidebar.title("🔎 Explainability")
user_id = user_selector()
st.header("Agent decision trace")

try:
    sessions = api.sessions(user_id)
except ApiError as exc:
    show_api_error(exc)
    st.stop()
if not sessions:
    st.info("No sessions for this user yet - chat first on the main page.")
    st.stop()

ids = [s["session_id"] for s in sessions]
current = st.session_state.session_id if st.session_state.session_id in ids else ids[0]
labels = {s["session_id"]: session_label(s) for s in sessions}
session_id = st.selectbox("Session", ids, index=ids.index(current), format_func=labels.get)
st.session_state.session_id = session_id

traces = api.decisions(user_id, session_id)
if not traces:
    st.info("This session has no answered turns yet.")
    st.stop()

# Model's current view of the conversation: shows summarization / pruning at work.
memory = api.history(user_id, session_id)
col1, col2, col3 = st.columns(3)
col1.metric("Turns", len(traces))
col2.metric("Messages in model context", len(memory["messages"]))
col3.metric("Summary words", len(memory["summary"].split()) if memory["summary"] else 0)

for trace in traces:  # newest first
    title = f"Turn {trace['turn']}: {trace['question'][:90]}  ·  {trace['latency_ms'] / 1000:.1f} s"
    with st.expander(title, expanded=trace is traces[0]):
        facts = {
            "Intent": trace["intent"],
            "Rewritten query": trace.get("rewritten_query") or "-",
            "Retrieval strategy": trace.get("retrieval_strategy") or "none",
            "Time to first token": f"{trace['first_token_ms']:.0f} ms" if trace.get("first_token_ms") else "-",
            "LangSmith trace id": trace.get("trace_id") or "-",
        }
        st.table(pd.DataFrame(facts.items(), columns=["", "value"]).set_index(""))

        decisions = trace["decisions"]
        st.markdown("**Agent timeline**")
        timeline = pd.DataFrame(
            [{"agent": d["agent"], "decision": d["decision"], "reasoning": d.get("reasoning", ""),
              "latency_ms": round(d["latency_ms"])} for d in decisions]
        )
        st.dataframe(timeline, hide_index=True, width="stretch")
        # One series (latency), ordered as executed: a single-hue horizontal bar chart needs no legend.
        st.bar_chart(timeline.set_index("agent")["latency_ms"], horizontal=True, y_label="", x_label="ms")

        retrieval = next((d for d in decisions if d["agent"] == "retrieval"), None)
        if retrieval and retrieval["details"].get("chunks"):
            st.markdown("**Retrieved chunks** (hybrid fusion of semantic + BM25 ranks)")
            chunks = pd.DataFrame(retrieval["details"]["chunks"])
            columns = [c for c in ("source", "section", "page", "score", "semantic_rank", "keyword_rank") if c in chunks]
            st.dataframe(chunks[columns], hide_index=True, width="stretch")
            if retrieval["details"].get("filter_relaxed"):
                st.warning("Router filters matched nothing - retrieval fell back to all documents.")

        memory_decision = next((d for d in decisions if d["agent"] == "memory_management"), None)
        if memory_decision and memory_decision["details"].get("stored"):
            st.success("Stored in long-term memory: " + "; ".join(memory_decision["details"]["stored"]))
