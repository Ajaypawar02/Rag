"""Performance metrics: latency, per-agent timings, routing mix and memory overhead."""

import pandas as pd
import streamlit as st

from api_client import ApiError
from common import get_client, init_state, show_api_error

st.set_page_config(page_title="Metrics", page_icon="📈", layout="wide")
init_state()
api = get_client()

st.header("Performance metrics")
st.caption("Aggregated over all answered turns (all users). Context preservation accuracy is measured offline "
           "with `python -m scripts.evaluate`.")
if st.button("↻ Refresh"):
    st.rerun()

try:
    m = api.metrics()
except ApiError as exc:
    show_api_error(exc)
    st.stop()

if not m["turns"]:
    st.info("No turns recorded yet - ask some questions first.")
    st.stop()


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value / 1000:.2f} s"


# Headline numbers are single values: stat tiles, not charts.
st.subheader("Latency")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Turns answered", m["turns"])
c2.metric("End-to-end p50", fmt_ms(m["latency_ms"]["p50"]))
c3.metric("End-to-end p95", fmt_ms(m["latency_ms"]["p95"]))
c4.metric("First token p50", fmt_ms(m["first_token_ms"]["p50"]), help="Streaming turns only")

left, right = st.columns(2)
with left:
    st.markdown("**Mean latency per agent**")
    agents = pd.Series(m["agent_latency_ms"], name="ms").sort_values(ascending=False)
    st.bar_chart(agents, horizontal=True, y_label="", x_label="ms")
    with st.expander("Table"):
        st.dataframe(agents.round(0).rename_axis("agent").reset_index(), hide_index=True)
with right:
    st.markdown("**Intent mix**")
    intents = pd.Series(m["intents"], name="turns").sort_values(ascending=False)
    st.bar_chart(intents, horizontal=True, y_label="", x_label="turns")
    with st.expander("Table"):
        st.dataframe(intents.rename_axis("intent").reset_index(), hide_index=True)

st.subheader("Retrieval")
r1, r2, r3 = st.columns(3)
r1.metric("Avg chunks per answer", m["avg_retrieved_chunks"] if m["avg_retrieved_chunks"] is not None else "-")
r2.metric("Citation rate", f"{m['citation_rate']:.0%}" if m["citation_rate"] is not None else "-",
          help="Share of document-backed answers that cite at least one source")
r3.metric("Strategies used", ", ".join(f"{k} ({v})" for k, v in m["retrieval_strategies"].items()))

st.subheader("Memory overhead (PostgreSQL)")
mem = m["memory"]
if "error" in mem:
    st.warning(f"Unavailable: {mem['error']}")
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Conversation state / session", f"{mem['state_kb_per_session']} KB")
    k2.metric("Checkpoints / session", mem["checkpoints_per_session"])
    k3.metric("Long-term memories", mem["long_term_memories"])
    k4.metric("Memory store size", f"{mem['store_bytes'] / 1024:.0f} KB")
    with st.expander("Raw values"):
        st.json(mem)
