"""
app.py — Milestone 5: Grounded Generation + Gradio Interface.

Wires the retriever (embed_retrieve.retrieve_context) to a Groq-hosted
llama-3.3-70b-versatile LLM with a strict grounding prompt, then exposes
it through a Gradio UI with separate answer and source-citation boxes.

Run:
    python app.py
    # opens http://localhost:7860
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from groq import Groq

from embed_retrieve import build_index, retrieve_context

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful food guide for UC Davis students. "
    "Answer using ONLY the information provided in the context below. "
    "Do not use any outside knowledge or make assumptions beyond what is written. "
    "If the context does not contain enough information to answer the question, "
    "respond with exactly: \"I don't have enough information on that.\""
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
# RAG pipeline
# --------------------------------------------------------------------------- #


def _format_context(results: list[dict]) -> str:
    """Serialize retrieved chunks into a numbered context block for the LLM."""
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['text'].strip()}")
    return "\n\n".join(lines)


def _format_citations(results: list[dict]) -> str:
    """Build a human-readable citation list from chunk metadata."""
    lines = []
    for i, r in enumerate(results, start=1):
        m = r["metadata"]
        source = m.get("source_file", "unknown")
        section = m.get("section") or m.get("platform") or ""
        dist = r["distance"]
        detail = f" — {section}" if section else ""
        lines.append(f"[{i}] {source}{detail}  (distance: {dist:.3f})")
    return "\n".join(lines)


def answer_question(query: str) -> tuple[str, str]:
    """Full RAG pipeline: retrieve → ground → generate.

    Returns (answer, citations).
    """
    if not query.strip():
        return "", ""

    results = retrieve_context(query, k=TOP_K)
    context_block = _format_context(results)

    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

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
    return answer, citations


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="The Unofficial Guide — UC Davis Food") as demo:
        gr.Markdown("# The Unofficial Guide\n**Ask anything about food at UC Davis.**")

        with gr.Row():
            query_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What are the hours for Segundo on a Tuesday?",
                lines=2,
            )

        submit_btn = gr.Button("Ask", variant="primary")

        with gr.Row():
            answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
            sources_box = gr.Textbox(label="Sources", lines=8, interactive=False)

        submit_btn.click(
            fn=answer_question,
            inputs=[query_box],
            outputs=[answer_box, sources_box],
        )
        query_box.submit(
            fn=answer_question,
            inputs=[query_box],
            outputs=[answer_box, sources_box],
        )

    return demo


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    build_index()
    build_ui().launch()
