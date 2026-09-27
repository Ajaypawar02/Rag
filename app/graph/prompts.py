"""Prompts for every agent, plus helpers that render state into prompt text.

Kept in one module so prompt wording can be reviewed and tuned in one place.
Prompts are provider-neutral: plain system + human messages, no vendor syntax.
"""

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage

MAX_ASSISTANT_CHARS_IN_HISTORY = 600  # long answers are truncated when shown as history


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def render_history(messages: list[AnyMessage], summary: str = "") -> str:
    """Conversation so far as plain text: rolling summary, then recent messages."""
    lines = []
    if summary:
        lines.append(f"Summary of earlier conversation:\n{summary}\n")
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.text}")
        elif isinstance(m, AIMessage):
            text = str(m.text)
            if len(text) > MAX_ASSISTANT_CHARS_IN_HISTORY:
                text = text[:MAX_ASSISTANT_CHARS_IN_HISTORY] + " ..."
            lines.append(f"Assistant: {text}")
    return "\n".join(lines) or "(no earlier conversation)"


MAX_CATALOG_ROWS = 50


def render_catalog(catalog: list[dict] | tuple[dict, ...]) -> str:
    """Available documentation as a compact table for routing decisions (newest uploads first)."""
    if not catalog:
        return "(no documents ingested)"
    uploads = sorted((d for d in catalog if d.get("origin") == "upload"),
                     key=lambda d: str(d.get("ingested_at") or ""), reverse=True)
    others = [d for d in catalog if d.get("origin") != "upload"]
    rows = ["source | title | product | version | origin | ingested_at"]
    rows += [
        f"{d['source']} | {d.get('title')} | {d.get('product') or '-'} | {d.get('version') or '-'} | "
        f"{d.get('origin') or '-'} | {d.get('ingested_at') or '-'}"
        for d in (uploads + others)[:MAX_CATALOG_ROWS]
    ]
    return "\n".join(rows)


def catalog_products(catalog: list[dict] | tuple[dict, ...]) -> list[str]:
    return sorted({d["product"] for d in catalog if d.get("product")})


def catalog_topics(catalog: list[dict] | tuple[dict, ...]) -> list[str]:
    """What the assistant can answer about: known products plus titles of documents without one (e.g. uploads)."""
    titles = sorted({d.get("title") or d["source"] for d in catalog if not d.get("product")})
    return catalog_products(catalog) + [f'"{t}"' for t in titles][:MAX_CATALOG_ROWS]


def render_context(retrieved: list[dict]) -> str:
    """Numbered documentation excerpts; the numbers are what answers cite as [n]."""
    blocks = []
    for i, chunk in enumerate(retrieved, start=1):
        location = f", page {chunk['page']}" if chunk.get("page") else ""
        uploaded = ", uploaded by the user" if chunk.get("origin") == "upload" else ""
        blocks.append(
            f"[{i}] (source: {chunk.get('source')} - \"{chunk.get('title')}\"{location}{uploaded})\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def history_before_latest(messages: list[AnyMessage]) -> list[AnyMessage]:
    """All messages except the trailing (current) user message."""
    if messages and isinstance(messages[-1], HumanMessage):
        return messages[:-1]
    return messages


# --------------------------------------------------------------------------- #
# Query Understanding
# --------------------------------------------------------------------------- #

UNDERSTANDING_SYSTEM = """You analyze the latest user message sent to a technical documentation assistant.

Classify the intent:
- new_question: a standalone technical question.
- follow_up: continues the conversation and relies on it ("how do I test it?", "and in v2?", "show an example").
- multi_part: asks two or more distinct questions at once ("How do X, and how do Y?").
- chitchat: greetings, thanks, small talk; nothing to look up.
- out_of_scope: clearly unrelated to software and to every document listed below (e.g. trivia, weather).

Questions about any listed document - including documents the user uploaded ("the report I uploaded",
"the file I shared") - are in scope. If unsure between out_of_scope and a question, choose the question:
a search that finds nothing is cheaper than refusing a valid question.

Set needs_history=true when the message cannot be understood on its own: pronouns or references
("it", "that", "the second one"), ellipsis ("and for Redis?"), or implied subjects.

Documentation available: {topics}

Give a one-sentence reasoning."""


def understanding_messages(query: str, history: list[AnyMessage], summary: str, topics: list[str]) -> list[BaseMessage]:
    return [
        SystemMessage(UNDERSTANDING_SYSTEM.format(topics=", ".join(topics) or "none")),
        HumanMessage(f"Conversation so far:\n{render_history(history, summary)}\n\nLatest user message:\n{query}"),
    ]


# --------------------------------------------------------------------------- #
# Query Rewriting
# --------------------------------------------------------------------------- #

REWRITING_SYSTEM = """You rewrite the user's latest message into a standalone search query for a documentation search engine.

- Resolve pronouns and references using the conversation ("it" -> the thing it refers to).
- Carry over the product, version and topic from the conversation when the message implies them.
- Keep identifiers, commands, error messages and version numbers exactly as written.
- Do not answer the question. Keep the query under 30 words.
- If the message is already standalone, return it unchanged.
{user_context}
Give a one-sentence reasoning."""

USER_CONTEXT_NOTE = """
Known facts about the user (use them only to fill in a product or version the user left unstated):
{memories}
"""


def _user_context(memories: list[str]) -> str:
    return USER_CONTEXT_NOTE.format(memories="\n".join(f"- {m}" for m in memories)) if memories else ""


def rewriting_messages(query: str, history: list[AnyMessage], summary: str, memories: list[str] = ()) -> list[BaseMessage]:
    return [
        SystemMessage(REWRITING_SYSTEM.format(user_context=_user_context(list(memories)))),
        HumanMessage(f"Conversation so far:\n{render_history(history, summary)}\n\nLatest user message:\n{query}"),
    ]


# --------------------------------------------------------------------------- #
# Query Decomposition
# --------------------------------------------------------------------------- #

DECOMPOSITION_SYSTEM = """You split a multi-part technical question into standalone sub-questions for documentation search.

- Produce 2 to 4 sub-questions, one per distinct information need.
- Each sub-question must make sense on its own: repeat the product name and context in each.
- If it is really a single question, return it as the only item.
{user_context}
Give a one-sentence reasoning."""


def decomposition_messages(query: str, memories: list[str] = ()) -> list[BaseMessage]:
    return [
        SystemMessage(DECOMPOSITION_SYSTEM.format(user_context=_user_context(list(memories)))),
        HumanMessage(query),
    ]


# --------------------------------------------------------------------------- #
# Retrieval Router
# --------------------------------------------------------------------------- #

ROUTER_SYSTEM = """You plan how to search a documentation knowledge base for a query.

Available documents:
{catalog}

Choose the search strategy:
- keyword: the query is dominated by exact tokens - function/class names, CLI commands, config keys, error messages.
- semantic: a conceptual or "how/why" question in natural language without exact identifiers.
- hybrid: mixed or unsure (the default).

Choose metadata filters (leave a filter null unless clearly justified):
- source: the exact source file name when the user asks about one specific document - by its title,
  or as "the document/report/file I uploaded" (then use the most recently ingested upload).
- product: only when the query clearly targets exactly one product from the list above.
- version: only when the user explicitly names a version that exists for that product.
- has_code: true only when the user asks for an example, code, snippet, command or config file.
{multi_part_note}
Give a one-sentence reasoning."""

MULTI_PART_NOTE = "\nThe query has several parts searched separately; set only filters that apply to every part:\n{parts}\n"


def router_messages(query: str, sub_queries: list[str], catalog: list[dict] | tuple[dict, ...]) -> list[BaseMessage]:
    note = MULTI_PART_NOTE.format(parts="\n".join(f"- {q}" for q in sub_queries)) if len(sub_queries) > 1 else ""
    return [
        SystemMessage(ROUTER_SYSTEM.format(catalog=render_catalog(catalog), multi_part_note=note)),
        HumanMessage(f"Query: {query}"),
    ]


# --------------------------------------------------------------------------- #
# Context Synthesis
# --------------------------------------------------------------------------- #

SYNTHESIS_SYSTEM = """You are a technical documentation assistant. Answer the user's latest message using the numbered documentation excerpts.

Rules:
- Base every technical statement on the excerpts and cite them inline as [1], [2], ...
- If the excerpts do not contain the answer, say so plainly. Do not invent APIs, flags or behavior.
- Excerpts marked "uploaded by the user" come from documents the user uploaded; when the user refers to
  "the report/document/file I uploaded", they mean those. For overview questions ("what is in it?"),
  summarize what those excerpts cover.
- Use the conversation for continuity: resolve what the user refers to and do not repeat what was already explained.
- Personalize with what you know about the user when relevant (their stack, versions, preferences),
  without mentioning that you remember things unless asked.
- Be concise: a direct answer first, then details. Include code from the excerpts when it helps.
- Format in Markdown.{relaxed_note}

What you know about this user:
{memories}

Documentation excerpts:
{context}"""

RELAXED_NOTE = (
    "\n- The documentation for the product/version the user asked about was not found; the excerpts come "
    "from other documents. Tell the user this before using them."
)

CHITCHAT_SYSTEM = """You are a friendly technical documentation assistant. Reply briefly (one to three sentences) and naturally
to the user's message. Acknowledge anything they tell you about themselves or their project.
Do not give technical explanations or code here - no documentation was looked up for this message;
invite them to ask their question instead. If it fits, mention that you can answer questions about: {products}.

What you know about this user:
{memories}"""

OUT_OF_SCOPE_SYSTEM = """You are a technical documentation assistant that only answers questions about: {products}.
The user's message is outside that scope. Politely say so in one or two sentences and suggest what you can help with.
Do not answer the out-of-scope question."""


def synthesis_messages(
    *,
    intent: str,
    history: list[AnyMessage],
    summary: str,
    query: str,
    retrieved: list[dict],
    memories: list[str],
    products: list[str],
    filter_relaxed: bool = False,
) -> list[BaseMessage]:
    memory_text = "\n".join(f"- {m}" for m in memories) or "(nothing yet)"
    product_text = ", ".join(products) or "the ingested documentation"
    if intent == "chitchat":
        system = CHITCHAT_SYSTEM.format(products=product_text, memories=memory_text)
    elif intent == "out_of_scope":
        system = OUT_OF_SCOPE_SYSTEM.format(products=product_text)
    else:
        system = SYNTHESIS_SYSTEM.format(
            memories=memory_text,
            context=render_context(retrieved) or "(no relevant excerpts were found)",
            relaxed_note=RELAXED_NOTE if filter_relaxed else "",
        )
    messages: list[BaseMessage] = [SystemMessage(system)]
    if summary:
        messages.append(SystemMessage(f"Summary of earlier conversation:\n{summary}"))
    messages.extend(history)
    messages.append(HumanMessage(query))
    return messages


# --------------------------------------------------------------------------- #
# Conversation Summarization
# --------------------------------------------------------------------------- #

SUMMARIZATION_SYSTEM = """You maintain a running summary of a conversation between a user and a technical documentation assistant.

Merge the existing summary with the new messages into one updated summary. Keep:
- the user's goals, environment, stack and versions;
- questions asked and the key answers (APIs, commands, settings, decisions);
- open issues or things the user still wants to do.
Drop greetings and filler. Write in the third person. Aim for about {target_words} words.
Output only the summary text."""


def summarization_messages(existing_summary: str, messages: list[AnyMessage], target_words: int) -> list[BaseMessage]:
    return [
        SystemMessage(SUMMARIZATION_SYSTEM.format(target_words=target_words)),
        HumanMessage(
            f"Existing summary:\n{existing_summary or '(none)'}\n\n"
            f"New messages to fold in:\n{render_history(messages)}"
        ),
    ]


# --------------------------------------------------------------------------- #
# Memory Management
# --------------------------------------------------------------------------- #

MEMORY_SYSTEM = """You decide what to remember long-term about a user of a technical documentation assistant,
based on their latest message.

Store only durable facts about the user that will help in future conversations:
- preference: how they like answers (e.g. "Prefers short answers with code examples")
- tech_stack: tools, frameworks, versions, platforms they use (e.g. "Uses Pydantic v2 with FastAPI")
- expertise: their experience level (e.g. "New to Kubernetes")
- goal: what they are building or trying to achieve (e.g. "Building a RAG chatbot")

Only store what the user explicitly states about themselves. Never infer facts from what they ask:
a question is not a fact about the user.
- "How do I override a dependency in tests?"      -> store nothing (not "Wants to override dependencies")
- "How do I run Redis with Docker Compose?"       -> store nothing (not "Uses Redis")
- "We deploy on EKS, how do I scale pods?"         -> store "Deploys on AWS EKS"
- "I'm new to Kubernetes, keep it simple"          -> store "New to Kubernetes", "Prefers simple explanations"

Never store: questions or tasks, facts about the documentation, one-off details, secrets, credentials
or personal identifiers. Do not repeat facts already known. Most messages contain nothing worth
storing - then return an empty list.

Already known about this user:
{known}"""


def memory_messages(query: str, known: list[str]) -> list[BaseMessage]:
    known_text = "\n".join(f"- {m}" for m in known) or "(nothing yet)"
    return [SystemMessage(MEMORY_SYSTEM.format(known=known_text)), HumanMessage(f"Latest user message:\n{query}")]
