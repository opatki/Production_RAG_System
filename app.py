"""
app.py — Milestone 5 + Stretch Features: Grounded Generation + Gradio Interface.

Core: wires retrieve_context → Groq llama-3.3-70b-versatile with a strict
grounding prompt, with two output boxes (answer + sources).

Stretch features:
  - Metadata filtering: dropdown to restrict retrieval to a single source file.
  - Conversational memory: last 3 turns are included in each prompt so
    follow-up questions can reference earlier context.

Run:
    python app.py
    # opens http://localhost:7860
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from groq import Groq

from embed_retrieve import build_index, retrieve_context, hybrid_retrieve

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
MEMORY_TURNS = 3  # number of prior turns to include in context

SOURCE_OPTIONS = [
    "All sources",
    "ucd_segundo_dc.txt",
    "gunrock.txt",
    "food_trucks.txt",
    "sage_street.txt",
    "latitude_market.txt",
    "latitude_restaurant.txt",
    "reddit.txt",
    "yelp.txt",
    "quora.txt",
    "tripadvisor.txt",
]

SYSTEM_PROMPT = (
    "You are a helpful food guide for UC Davis students. "
    "Answer using ONLY the information provided in the context below. "
    "Do not use any outside knowledge or make assumptions beyond what is written. "
    "If the context does not contain enough information to answer the question, "
    "respond with exactly: \"I don't have enough information on that.\""
)

REWRITE_PROMPT = (
    "Given the conversation history and a follow-up question, rewrite the follow-up "
    "into a fully self-contained search query with no pronouns or references like "
    "'it', 'there', or 'that place'. Output only the rewritten query, nothing else."
)

# --------------------------------------------------------------------------- #
# Groq client (reads GROQ_API_KEY from environment)
# --------------------------------------------------------------------------- #

_groq: Groq | None = None


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        _groq = Groq(api_key=api_key)
    return _groq


# --------------------------------------------------------------------------- #
# RAG pipeline helpers
# --------------------------------------------------------------------------- #


def _format_context(results: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {r['text'].strip()}" for i, r in enumerate(results, 1))


def _format_citations(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        source = m.get("source_file", "unknown")
        section = m.get("section") or m.get("platform") or ""
        dist = r.get("distance", float("nan"))
        detail = f" — {section}" if section else ""
        dist_str = f"{dist:.3f}" if dist == dist else "n/a"  # nan check
        lines.append(f"[{i}] {source}{detail}  (distance: {dist_str})")
    return "\n".join(lines)


def _format_memory(history: list[tuple[str, str]]) -> str:
    """Format the last MEMORY_TURNS turns as a conversation-history block."""
    if not history:
        return ""
    recent = history[-MEMORY_TURNS:]
    lines = ["\n\nConversation history (for context only — do not cite):"]
    for q, a in recent:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def _rewrite_query(query: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up query using conversation history for retrieval.

    Resolves pronouns like 'it', 'there', 'that place' so the retrieval step
    gets a self-contained query rather than a pronoun-heavy fragment.
    """
    if not history:
        return query
    hist_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-MEMORY_TURNS:])
    c = get_groq().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"History:\n{hist_text}\n\nFollow-up: {query}"},
        ],
        temperature=0,
    )
    return c.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Core answer function (used by both submit paths)
# --------------------------------------------------------------------------- #


def answer_question(
    query: str,
    source_filter: str,
    history: list[tuple[str, str]],
) -> tuple[str, str, list[tuple[str, str]]]:
    """Full RAG pipeline with optional metadata filtering and conversational memory.

    Returns (answer, citations, updated_history).
    """
    if not query.strip():
        return "", "", history

    where = None if source_filter == "All sources" else {"source_file": source_filter}
    retrieval_query = _rewrite_query(query, history)
    results = retrieve_context(retrieval_query, k=TOP_K, where=where)

    context_block = _format_context(results)
    memory_block = _format_memory(history)
    user_message = f"Context:\n{context_block}{memory_block}\n\nQuestion: {query}"

    completion = get_groq().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content.strip()
    citations = _format_citations(results)
    updated_history = history + [(query, answer)]
    return answer, citations, updated_history


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="The Unofficial Guide — UC Davis Food") as demo:
        gr.Markdown("# The Unofficial Guide 🍕🍽️\n**Ask anything about food at UC Davis.**")

        history_state = gr.State([])

        with gr.Row():
            query_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What are the hours for Segundo on a Tuesday?",
                lines=2,
                scale=3,
            )
            source_dropdown = gr.Dropdown(
                label="Filter by source (optional)",
                choices=SOURCE_OPTIONS,
                value="All sources",
                scale=1,
            )

        with gr.Row():
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear history")

        with gr.Row():
            answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
            sources_box = gr.Textbox(label="Sources", lines=8, interactive=False)

        history_box = gr.Textbox(
            label="Conversation history",
            lines=10,
            interactive=False,
        )

        def _history_text(history: list[tuple[str, str]]) -> str:
            if not history:
                return ""
            lines = []
            for q, a in history:
                lines.append(f"You: {q}")
                lines.append(f"Assistant: {a}")
                lines.append("")
            return "\n".join(lines).strip()

        def _submit(query, source_filter, history):
            answer, citations, updated = answer_question(query, source_filter, history)
            return answer, citations, updated, _history_text(updated)

        submit_btn.click(
            fn=_submit,
            inputs=[query_box, source_dropdown, history_state],
            outputs=[answer_box, sources_box, history_state, history_box],
        )
        query_box.submit(
            fn=_submit,
            inputs=[query_box, source_dropdown, history_state],
            outputs=[answer_box, sources_box, history_state, history_box],
        )
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[history_state, history_box],
        )

    return demo


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    build_index()
    build_ui().launch()
