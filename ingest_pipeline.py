"""
ingest_pipeline.py — Milestone 3: Document Pipeline for "The Unofficial Guide" RAG system.

Loads the pre-scraped .txt files in ./documents, cleans them, and produces chunks
using a HYBRID strategy keyed on the source type (see planning.md > Chunking Strategy):

    Track A — Official UCD Dining pages (structured):
        food_trucks, gunrock, latitude_market, latitude_restaurant,
        sage_street, ucd_segundo_dc
        -> Split on STRUCTURAL HEADERS, 0 overlap.
        -> Inject global page context (restaurant name + source URL) into METADATA.

    Track B — Social / unstructured (conversational):
        quora, reddit, tripadvisor, yelp
        -> Sliding window, 500-800 chars, 100-char overlap.
        -> PREPEND the source platform into the TEXT BODY of every chunk so the
           embedding captures institutional/spatial context (planning.md).

IMPORTANT NOTE ON "MARKDOWN" CHUNKING
-------------------------------------
The planning doc specifies a "markdown-element" chunker that splits on headers like
`### Hours`. The scraped source files contain NO markdown headers — they are flat
text where section labels (`Hours`, `Menu`, `Contact Us`, ...) appear as bare lines.
A literal markdown splitter would emit one chunk per file. Track A therefore uses a
header-DETECTION heuristic that recognizes those bare label lines as structural
headers and splits on them — preserving the spec's intent (keep hours/menu/payment
facts bundled as one semantic unit) on the data we actually have.

This script is the VALIDATION CHECKPOINT only. It does not embed anything.
The `build_corpus()` function returns the chunk list (text + metadata) that
Milestone 4 will feed to all-MiniLM-L6-v2 / ChromaDB.

Run:
    python ingest_pipeline.py
"""

from __future__ import annotations

import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# The source docs are UTF-8 (ø, —, é, etc.). Force UTF-8 stdout so the validation
# printout renders correctly even on a default cp1252 Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Track A: official UCD pages -> structural-header chunking, context in metadata.
TRACK_A_STEMS = {
    "food_trucks",
    "gunrock",
    "latitude_market",
    "latitude_restaurant",
    "sage_street",
    "ucd_segundo_dc",
}

# Track B: social / unstructured -> sliding window, context prepended to body.
TRACK_B_STEMS = {"quora", "reddit", "tripadvisor", "yelp"}

# Human-readable global context per source file (injected per the spec).
DISPLAY_NAMES = {
    "food_trucks": "UC Davis Food Trucks",
    "gunrock": "The Gunrock (featuring Sudwerk Brewing Co.)",
    "latitude_market": "Latitude Market",
    "latitude_restaurant": "Latitude Restaurant",
    "sage_street": "Sage Street Market | Cafe",
    "ucd_segundo_dc": "Segundo Dining Commons",
    "quora": "Quora",
    "reddit": "Reddit (r/UCDavis)",
    "tripadvisor": "TripAdvisor",
    "yelp": "Yelp",
}

# Sliding-window parameters for Track B (planning.md: 500-800 chars, 100 overlap).
WINDOW_MIN = 500
WINDOW_MAX = 800
WINDOW_OVERLAP = 100

# Track A: cap any single header-section so giant menus stay embeddable.
# Sub-splits inherit the same header/metadata and keep 0 overlap (spec-compliant).
MAX_SECTION_CHARS = 1200

# Reproducible "random" inspection sample. Set to None for true randomness.
RANDOM_SEED = None
SAMPLE_SIZE = 5

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

# Boilerplate / navigation lines that recur across the official UCD pages.
# Matched case-insensitively against a stripped line (exact match).
_BOILERPLATE_LINES = {
    "sign up today for",
    "the davis dish",
    "newsletter!",
    "sign up here for",
    "!",
    "close",
}

# Everything from one of these lines to end-of-file is footer/nav cruft.
_FOOTER_SENTINELS = {
    "you belong here",
    "quick links",
}

# Lines that are pure vote counts / forum chrome (Track B light cleaning).
_FORUM_NOISE = re.compile(r"^[•·∙\-\s]*$|^\d+$")


def clean_text(text: str) -> str:
    """Lightweight whitespace / formatting normalization applied to BOTH tracks.

    - Normalizes non-breaking spaces and tabs to plain spaces.
    - Collapses runs of intra-line whitespace.
    - Strips trailing whitespace from every line.
    - Collapses 3+ blank lines down to a single blank line.
    """
    text = text.replace(" ", " ").replace("\t", " ")
    lines = [re.sub(r"[ ]{2,}", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_official_boilerplate(text: str) -> str:
    """Drop newsletter intro chrome and the repeated footer/nav block (Track A)."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped in _FOOTER_SENTINELS:
            break  # everything after this is footer cruft
        if stripped in _BOILERPLATE_LINES:
            continue
        out.append(line)
    return "\n".join(out).strip()


def _clean_forum_text(text: str) -> str:
    """Drop standalone vote-count / bullet-only lines from social sources (Track B)."""
    kept = [ln for ln in text.splitlines() if not _FORUM_NOISE.match(ln.strip())]
    return "\n".join(kept).strip()


# --------------------------------------------------------------------------- #
# Chunk model
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """A single retrievable unit. `text` is embedded; `metadata` rides along."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def char_len(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------- #
# Track A — structural-header chunker (official UCD pages)
# --------------------------------------------------------------------------- #

# High-confidence canonical section labels seen across the UCD dining pages.
_KNOWN_HEADERS = {
    "menu",
    "restaurant menu",
    "special menu",
    "hours",
    "regular hours",
    "academic year hours",
    "daily schedule",
    "contact us",
    "contact",
    "location",
    "reservations",
    "prices at the door",
    "special meals & events",
    "overview",
}


def _merge_split_kv(lines: list[str]) -> list[str]:
    """Re-join the scraped split key/value layout into single `Label: value` lines.

    The pages store data as a label line followed by a `:`-prefixed value line::

        Calories
        : 254.02            ->   Calories: 254.02

    Empty-value rows (e.g. an unpopulated `Serving Size :`) are dropped as noise.
    This is what stops nutrition tables from exploding into one chunk per nutrient.
    """
    merged: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        cur = lines[i].strip()
        if not cur:
            i += 1
            continue
        nxt = lines[i + 1].strip() if i + 1 < n else ""
        if nxt.startswith(":"):
            value = nxt[1:].strip()
            if value:
                merged.append(f"{cur}: {value}")
            # else: empty value -> drop the row entirely
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged


# Trailing words that mean the previous line left a sentence dangling, so the next
# capitalized line is a continuation (`...credit cards and` / `AggieCash`), not a header.
_DANGLING_CONNECTIVES = {
    "and", "or", "with", "the", "a", "an", "of", "to", "for", "at", "in", "on", "&",
}


def _looks_like_header(line: str, prev_line: str = "") -> bool:
    """Heuristic: does this bare line act as a structural section header?

    The scraped pages have no markdown, so we infer headers. A header is a short,
    Title/Capitalized label with no colon and no terminal sentence punctuation —
    e.g. `Hours`, `Contact Us`, `Bangin' Bowls`, a dish name. After `_merge_split_kv`,
    every data row (`Calories: 254.02`, `Monday–Friday: 7 AM–10 PM`) contains a
    colon and is therefore body, so it bundles under its dish/section header.

    `prev_line` guards against scraped sentence-breaks: if the previous prose line
    ends with a dangling connective, this line continues that sentence (body).
    """
    s = line.strip()
    if not s:
        return False
    if ":" in s:
        return False  # post-merge data row (label: value) -> body, never a header
    if s.lower() in _KNOWN_HEADERS:
        return True
    if len(s) > 50 or len(s.split()) > 7:
        return False
    if not s[0].isalpha() or s[0].islower():
        return False  # body sentences / prices / numbers / lowercase fragments
    if s[-1] in ".,;!":
        return False  # full sentences end in punctuation; headers don't
    if "," in s:
        return False  # excludes dated schedule lines, list-y sentences
    if "@" in s or "http" in s.lower() or re.search(r"\d{3}-\d{3,4}", s):
        return False  # emails / urls / phone numbers

    p = prev_line.strip()
    if p and ":" not in p:  # only prose lines can dangle (data rows have colons)
        last = re.sub(r"[^\w&]", "", p.split()[-1]).lower() if p.split() else ""
        if last in _DANGLING_CONNECTIVES:
            return False  # mid-sentence continuation, not a header
    return True


def _split_oversized(body: str) -> list[str]:
    """Split a too-large section body on paragraph/line boundaries, 0 overlap."""
    if len(body) <= MAX_SECTION_CHARS:
        return [body]
    pieces, current = [], ""
    for para in re.split(r"\n", body):
        candidate = f"{current}\n{para}".strip() if current else para
        if len(candidate) > MAX_SECTION_CHARS and current:
            pieces.append(current.strip())
            current = para
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_track_a(raw: str, stem: str) -> list[Chunk]:
    """Header-element chunker: split on detected section headers, 0 overlap.

    Global page context (restaurant name + source URL) is injected into METADATA,
    not the body, per the planning spec.
    """
    # Pull the leading `Source URL:` line for provenance, then drop it from content.
    source_url = ""
    lines = raw.splitlines()
    if lines and lines[0].lower().startswith("source url:"):
        source_url = lines[0].split(":", 1)[1].strip()
        lines = lines[1:]

    body_text = _strip_official_boilerplate("\n".join(lines))
    restaurant = DISPLAY_NAMES.get(stem, stem)

    # Re-join the scraped split key/value layout before detecting headers.
    normalized = _merge_split_kv(body_text.splitlines())

    # Group lines into (header, body) sections.
    sections: list[tuple[str, list[str]]] = []
    current_header = "Overview"
    current_body: list[str] = []
    prev_line = ""
    for line in normalized:
        if not line.strip():
            continue
        if _looks_like_header(line, prev_line):
            if current_body:
                sections.append((current_header, current_body))
            current_header = line.strip()
            current_body = []
        else:
            current_body.append(line.strip())
        prev_line = line
    if current_body:
        sections.append((current_header, current_body))

    chunks: list[Chunk] = []
    for header, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for piece in _split_oversized(body):
            # The header is naturally the first line of the chunk text.
            text = f"{header}\n{piece}".strip()
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        "source_file": f"{stem}.txt",
                        "source_url": source_url,
                        "restaurant": restaurant,   # <-- global context (metadata)
                        "section": header,
                        "track": "A",
                        "strategy": "structural-header",
                        "char_len": len(text),
                    },
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Track B — sliding-window chunker (social / unstructured)
# --------------------------------------------------------------------------- #


def _find_break(window: str, min_size: int) -> int | None:
    """Return a 'nice' break offset (>= min_size) inside window, else None.

    Prefers a sentence boundary; falls back to the last whitespace. Keeps chunks
    from cutting mid-word/mid-sentence.
    """
    tail = window[min_size:]
    # Prefer the last sentence terminator in the tail region.
    sentence = list(re.finditer(r"[.!?]\s", tail))
    if sentence:
        return min_size + sentence[-1].end()
    space = tail.rfind(" ")
    if space != -1:
        return min_size + space + 1
    return None


def chunk_track_b(raw: str, stem: str) -> list[Chunk]:
    """Sliding window, 500-800 chars, 100-char overlap.

    The platform/source label is PREPENDED to the body of every chunk so the
    embedding captures context even when the raw text uses pronouns (planning.md).
    """
    text = _clean_forum_text(raw)
    text = re.sub(r"\s+", " ", text).strip()  # flatten to a single stream
    platform = DISPLAY_NAMES.get(stem, stem)
    context_prefix = f"[Source: {platform}] "

    chunks: list[Chunk] = []
    start, n, idx = 0, len(text), 0
    step = WINDOW_MAX - WINDOW_OVERLAP
    while start < n:
        end = min(start + WINDOW_MAX, n)
        if end < n:  # not the final window -> try to land on a clean boundary
            brk = _find_break(text[start:end], WINDOW_MIN)
            if brk is not None:
                end = start + brk
        body = text[start:end].strip()
        if body:
            full = f"{context_prefix}{body}"
            chunks.append(
                Chunk(
                    text=full,
                    metadata={
                        "source_file": f"{stem}.txt",
                        "platform": platform,
                        "track": "B",
                        "strategy": "sliding-window",
                        "window": f"{WINDOW_MIN}-{WINDOW_MAX}",
                        "overlap": WINDOW_OVERLAP,
                        "char_len": len(full),
                        "chunk_index": idx,
                    },
                )
            )
            idx += 1
        if end >= n:
            break
        start = max(end - WINDOW_OVERLAP, start + 1)  # advance with overlap; never stall
    return chunks


# --------------------------------------------------------------------------- #
# Loading + orchestration
# --------------------------------------------------------------------------- #


def load_documents(directory: Path = DOCUMENTS_DIR) -> dict[str, str]:
    """Load every .txt file in `directory` -> {stem: cleaned_text}."""
    if not directory.is_dir():
        raise FileNotFoundError(f"documents directory not found: {directory}")
    docs: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        docs[path.stem] = clean_text(raw)
    if not docs:
        raise FileNotFoundError(f"no .txt files found in {directory}")
    return docs


def build_corpus(directory: Path = DOCUMENTS_DIR) -> list[Chunk]:
    """Full pipeline: load -> clean -> route to Track A/B -> return all chunks.

    This is the importable entry point Milestone 4 will consume.
    """
    docs = load_documents(directory)
    corpus: list[Chunk] = []
    for stem, text in docs.items():
        if stem in TRACK_A_STEMS:
            corpus.extend(chunk_track_a(text, stem))
        elif stem in TRACK_B_STEMS:
            corpus.extend(chunk_track_b(text, stem))
        else:
            # Unknown source: default to the safer unstructured chunker + warn.
            print(f"[warn] '{stem}.txt' not classified; defaulting to Track B.")
            corpus.extend(chunk_track_b(text, stem))
    return corpus


# --------------------------------------------------------------------------- #
# Validation checkpoint (NO embedding — visual inspection only)
# --------------------------------------------------------------------------- #


def _print_chunk(chunk: Chunk, ordinal: int) -> None:
    print(f"\n{'=' * 78}\nCHUNK #{ordinal}  |  {chunk.char_len} chars")
    print("-" * 78)
    for key, value in chunk.metadata.items():
        print(f"  {key:<12}: {value}")
    print("-" * 78)
    print(textwrap.fill(chunk.text, width=78, replace_whitespace=False))
    print("=" * 78)


def validate(corpus: list[Chunk]) -> None:
    """Count chunks, report distribution, and print SAMPLE_SIZE random chunks."""
    total = len(corpus)
    track_counts: dict[str, int] = {}
    file_counts: dict[str, int] = {}
    char_lens: list[int] = []
    for c in corpus:
        track_counts[c.metadata["track"]] = track_counts.get(c.metadata["track"], 0) + 1
        f = c.metadata["source_file"]
        file_counts[f] = file_counts.get(f, 0) + 1
        char_lens.append(c.char_len)

    print("\n" + "#" * 78)
    print("# VALIDATION CHECKPOINT — Milestone 3 (no embedding performed)")
    print("#" * 78)
    print(f"\nTotal chunks: {total}")
    print(f"  Track A (structural-header): {track_counts.get('A', 0)}")
    print(f"  Track B (sliding-window)   : {track_counts.get('B', 0)}")
    if char_lens:
        print(
            f"  Chunk size  : min={min(char_lens)}  "
            f"max={max(char_lens)}  avg={sum(char_lens) // len(char_lens)} chars"
        )

    print("\nChunks per source file:")
    for f in sorted(file_counts):
        print(f"  {f:<26}: {file_counts[f]}")

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    k = min(SAMPLE_SIZE, total)
    print(f"\nRandomly sampling {k} complete chunks for visual inspection:")
    for i, chunk in enumerate(random.sample(corpus, k), start=1):
        _print_chunk(chunk, i)


if __name__ == "__main__":
    corpus = build_corpus()
    validate(corpus)
