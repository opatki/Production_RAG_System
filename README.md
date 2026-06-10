# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

For my system, I want my domain to be "Food at UC Davis." Throughout my time at UC Davis, I found it hard to find places that were affordable and satisying as freshman. We were given $200 in AggieCash and I spent so much of it on this 1 food truck because I didn't know which other popular food spots existed. This knowledge is valuable because everyone has to eat and the dining commons often don't satisfy the needs of all students at UC Davis. This tool would be incredibly useful in answering student questions and help them find exactly the meal they are looking for. 


---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | UCD Dining Website | URL | [Segundo Dining Commons](https://housing.ucdavis.edu/dining/dining-commons/segundo/) |
| 2 | UCD Dining Website | URL | [Gunrock](https://housing.ucdavis.edu/dining/the-gunrock/) |
| 3 | UCD Dining Website | URL | [Food Trucks](https://housing.ucdavis.edu/dining/food-trucks/) |
| 4 | UCD Dining Website | URL | [Sage Street](https://housing.ucdavis.edu/dining/sage-street/) |
| 5 | UCD Dining Website | URL | [Latitude Market](https://housing.ucdavis.edu/dining/latitude-market/) |
| 6 | UCD Dining Website | URL | [Latitude](https://housing.ucdavis.edu/dining/latitude/) |
| 7 | Reddit | URL | [Restaurants to Try before You Graduate](https://www.reddit.com/r/UCDavis/comments/1j2xry7/must_try_davis_restaurants_before_you_graduate/) |
| 8 | Yelp | URL | [Nice Restaurants in Davis](https://www.yelp.com/search?find_desc=Nice+Restaurant&find_loc=Davis%2C+CA) |
| 9 | Tripadvisor | URL | [Best Places in Davis](https://www.tripadvisor.com/Restaurants-g32283-Davis_California.html) |
| 10 | Quora | URL | [Best Bite in Davis](https://www.quora.com/What-are-the-best-restaurants-near-UC-Daviss-campus) |

---

## Chunking Strategy

**Chunk size:** 
I will use a Hybrid Chunking Strategy depending on the source type, with an average target size of 500–800 characters (~125–200 tokens) for unstructured reviews, and Markdown Element-Based Chunking for official UC Davis web pages.

**Overlap:**
For unstructured data (Reddit, Yelp, Quora, Tripadvisor), I will use a 100-character (~25 tokens) sliding window overlap. For official UCD Dining pages chunked by Markdown headers, I will use 0 character overlap, but explicitly inject the global page context (e.g., Restaurant: Segundo Dining Commons) into the metadata of every sub-chunk.

**Why these choices fit your documents:**
My dataset contains two distinct document archetypes that require different handling to prevent context fragmentation:
- Official UCD Dining Pages (Structured): These pages contain dense, tabular, or highly categorized information (e.g., operational hours, explicit menu items, AggieCash acceptance). Fixed-token chunking would inevitably split a restaurant’s name from its operational hours or accepted payment methods. Chunking strictly by Markdown elements (headers like ### Hours or ### Menu) guarantees that critical business facts stay bundled together as a single semantic unit.
- Social Media/Review Platforms (Unstructured): Reddit threads and Yelp reviews are highly conversational. A single Reddit comment might list five different favorite spots in Davis in a single paragraph. A 500–800 character size is tight enough to isolate individual restaurant recommendations without mashing distinct spots together, while the 100-character overlap ensures that transitions or multi-sentence descriptions of a specific dish (like a review highlighting a specific spicy wing flavor or garlic knot spot) aren't cut in half.
- Metadata Injection (Context Preservation): Because forum comments often use pronouns (e.g., "The food truck outside Silo has the best sliders, I spent all my AggieCash there"), I will preprocess unstructured files by appending the thread title or platform metadata directly into the text body of each chunk. This ensures the vector embeddings capture the spatial and institutional context of Davis.

**Final chunk count:** 
992 Chunks

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**
`all-MiniLM-L6-v2` (HuggingFace sentence-transformers). It generates 384-dimensional vectors and runs entirely locally, requiring no API calls or cost during development. It is lightweight and fast, making it well-suited for iterating on a local Chroma vector store.

**Production tradeoff reflection:**
If deploying this to a broader student base and cost wasn't an issue, I would upgrade to OpenAI's `text-embedding-3-small`. While `all-MiniLM-L6-v2` is fast and free, it struggles slightly with highly specific domain jargon or slang (e.g., students calling Segundo "the DC" or referring to AggieCash colloquially). A commercial model offers a longer context window and better multilingual support, which is useful if international students are searching for specific hometown cuisines. However, it introduces per-query API latency and ongoing cost, requiring a tradeoff analysis against backend throughput requirements and budget constraints.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
The system prompt contains two grounding mechanisms working together. First, a hard constraint in the system role: *"Answer using ONLY the information provided in the context below. Do not use any outside knowledge or make assumptions beyond what is written. If the context does not contain enough information to answer the question, respond with exactly: 'I don't have enough information on that.'"* Second, a structural constraint in the user message: the top-5 retrieved chunks are serialized as a numbered context block (`[1] … [2] … [3] …`) and prepended to the question, so the model's attention is explicitly anchored to the retrieved material before it sees the query. Together, the role-level instruction tells the model *what rules to follow*, and the message-level formatting tells it *exactly what text it is allowed to draw from*.

**How source attribution is surfaced in the response:**
The Gradio interface uses two independent output boxes. The left box contains only the LLM-generated answer. The right box ("Sources") is populated programmatically by `_format_citations()` in [app.py](app.py), which reads the `source_file`, `section`/`platform`, and cosine distance from the ChromaDB metadata of each retrieved chunk — it never asks the LLM to cite anything. This means citations are always accurate and tied to actual retrieved chunks, regardless of what the model generates.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**

**What the system returned:**

**Root cause (tied to a specific pipeline stage):**

**What you would change to fix it:**

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

**One way your implementation diverged from the spec, and why:**

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*

**Instance 2**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*
