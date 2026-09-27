"""Chat with the assistant in the terminal (full LangGraph pipeline, persisted in Postgres).

Usage:
    python -m scripts.chat                          # new session for user "demo"
    python -m scripts.chat --user alice --session my-session   # resume a session
    python -m scripts.chat --quiet                  # hide the agent decision trace

Type /new for a new session, /quit to exit.
"""

import argparse
import logging
import sys
import uuid

from app.graph.graph import get_graph, run_turn
from app.graph.state import turn_timings


def print_turn(state: dict, quiet: bool) -> None:
    print(f"\nassistant> {state['answer']}\n")
    for s in state.get("sources", []):
        where = s.get("section") or (f"page {s['page']}" if s.get("page") else "")
        print(f"  [{s['index']}] {s.get('source')}  {where}")
    if quiet:
        return
    print("\n  -- agent trace --")
    for d in state.get("decisions", []):
        print(f"  {d['agent']:<20} {d['latency_ms']:>7.0f} ms  {d['decision']}")
        if d.get("reasoning"):
            print(f"  {'':<31}{d['reasoning'][:110]}")
    print(f"  {'total':<20} {sum(turn_timings(state).values()):>7.0f} ms\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default="demo")
    parser.add_argument("--session", default=None, help="session/thread id to resume (default: new)")
    parser.add_argument("--quiet", action="store_true", help="hide the agent decision trace")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    graph = get_graph()
    session = args.session or uuid.uuid4().hex[:12]
    print(f"user={args.user} session={session}  (/new, /quit)")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message == "/quit":
            return 0
        if message == "/new":
            session = uuid.uuid4().hex[:12]
            print(f"new session={session}")
            continue
        print_turn(run_turn(graph, message, args.user, session), args.quiet)


if __name__ == "__main__":
    sys.exit(main())
