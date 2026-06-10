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

**Sample chunks:**

> **Chunk 1 — `ucd_segundo_dc.txt` (Track A — structural header)**
> ```
> Segundo Dining Commons — Academic Year Hours
> Tuesday, September 23, 2025 – Thursday, June 11, 2026.
> Monday–Friday: 7 AM–10 PM
> Saturday: 9 AM–8 PM
> Sunday: 9 AM–8 PM
> Holidays: 9 AM–8 PM
> ```

> **Chunk 2 — `gunrock.txt` (Track A — structural header)**
> ```
> The Gunrock (featuring Sudwerk Brewing Co.) — Reservations
> The Gunrock is now offering reservations in the dining room
> to host your next meeting or gathering.
> ```

> **Chunk 3 — `quora.txt` (Track B — sliding window)**
> ```
> [Source: Quora] good chai and a decent buffet. Dixon's got a fantastic
> Punjabi Dhaba (also the restaurant's name). I will never sink to call
> Raja's Tandoor actual Indian food. They call butter chicken "tikka masala."
> TIKKA MASALA. My Indian eyes cry. Vietnamese - Pho King 4 (lol, I know
> how that's pronounced)...
> ```

> **Chunk 4 — `reddit.txt` (Track B — sliding window)**
> ```
> [Source: Reddit (r/UCDavis)] the smoothie truck on campus and a place
> downtown called open rice everything is like less than $10! Tacos
> Guadalajara, Tasty Kitchen, Tim's Hawaiian, and Yuchan Shokudo.
> All go-to's whenever my family comes to visit!
> ```

> **Chunk 5 — `yelp.txt` (Track B — sliding window)**
> ```
> [Source: Yelp] "Amazing place, beautiful ambiance, great service, and
> very good staff." Japanese. 88 BaoBao 4.3 (48 reviews) Davis — "Nice
> restaurant near UC Davis, walking distance from Yosemite dorm and Ulta.
> We are greeted by..." Dim Sum, Noodles, Asian Fusion.
> ```

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

## Retrieval Examples

**Example 1 — Query:** "What are the hours for Segundo Dining Commons on a Tuesday?"

| Rank | Source | Section | Distance |
|------|--------|---------|----------|
| 1 | `ucd_segundo_dc.txt` | Academic Year Hours | 0.170 |
| 2 | `ucd_segundo_dc.txt` | Lunch | 0.222 |
| 3 | `ucd_segundo_dc.txt` | Lunch | 0.222 |

*Why these chunks are relevant:* Chunk 1 is an exact semantic match — it contains the phrase "Monday–Friday: 7 AM–10 PM" alongside the date range for the academic year, which directly answers the question. The distance of 0.170 is well below the 0.5 on-topic threshold, signaling a high-confidence hit. Chunks 2 and 3 are from the same source and contain station-level lunch schedules, which are topically adjacent (time/hours at Segundo) even though they don't answer the specific Tuesday hours question. All three results coming from the same source file confirms the retriever correctly scoped to the right document.

---

**Example 2 — Query:** "Does The Gunrock accept meal swipes or AggieCash?"

| Rank | Source | Section | Distance |
|------|--------|---------|----------|
| 1 | `gunrock.txt` | Reservations | 0.436 |
| 2 | `gunrock.txt` | View More Special Meals & Events | 0.440 |
| 3 | `ucd_segundo_dc.txt` | Aggie Swipe Plus Meal Plans | 0.452 |

*Why these chunks are partially relevant:* Chunks 1 and 2 are from the correct source file (`gunrock.txt`), so the retriever identified the right restaurant. However, neither contains payment policy information — they cover reservations and promotional events. Chunk 3 is from a different restaurant (Segundo) and was retrieved because the terms "AggieCash" and "meal swipes" appear frequently in its text, creating a false-positive term match. All three distances are above 0.43, which is why the system correctly returned the fallback rather than hallucinating an answer.

---

**Example 3 — Query:** "Are there any places near the dorms that serve Indian street food like pav bhaji?"

| Rank | Source | Section | Distance |
|------|--------|---------|----------|
| 1 | `quora.txt` | Quora | 0.457 |
| 2 | `quora.txt` | Quora | 0.459 |
| 3 | `tripadvisor.txt` | TripAdvisor | 0.463 |

*Why these chunks are partially relevant:* Chunk 2 contains mentions of Indian cuisine ("good chai," "Punjabi Dhaba," "Raja's Tandoor," "butter chicken"), making it semantically adjacent to the query's Indian food intent. However, it does not mention pav bhaji specifically or proximity to dorms. Chunk 1 discusses general campus budget spots (Dos Coyotes) with no Indian food connection. Chunk 3 is off-topic entirely. The distances (0.457–0.463) are all above the on-topic threshold, reflecting that no chunk in the corpus closely matches this specific query.

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

**Example response 1 — on-topic query:**

> **Query:** What are the hours for Segundo Dining Commons on a Tuesday?
>
> **Answer box:** The hours for Segundo Dining Commons on a Tuesday are 7 AM–10 PM.
>
> **Sources box:**
> ```
> [1] ucd_segundo_dc.txt — Academic Year Hours  (distance: 0.170)
> [2] ucd_segundo_dc.txt — Lunch                (distance: 0.222)
> [3] ucd_segundo_dc.txt — Lunch                (distance: 0.222)
> [4] ucd_segundo_dc.txt — Lunch                (distance: 0.223)
> [5] ucd_segundo_dc.txt — Lunch                (distance: 0.223)
> ```

**Example response 2 — partial retrieval, grounding fallback:**

> **Query:** Does The Gunrock accept meal swipes or AggieCash?
>
> **Answer box:** I don't have enough information on that.
>
> **Sources box:**
> ```
> [1] gunrock.txt — Reservations                      (distance: 0.436)
> [2] gunrock.txt — View More Special Meals & Events  (distance: 0.440)
> [3] ucd_segundo_dc.txt — Aggie Swipe Plus Meal Plans (distance: 0.452)
> [4] gunrock.txt — We're Hungry for Feedback         (distance: 0.453)
> [5] gunrock.txt — The Gunrock                       (distance: 0.492)
> ```

**Out-of-scope query — refusal:**

> **Query:** Where can I buy a parking permit on campus?
>
> **Answer box:** I don't have enough information on that.
>
> **Sources box:**
> ```
> [1] food_trucks.txt — Food trucks accept credit cards and  (distance: 0.611)
> [2] food_trucks.txt — Send Us Feedback                     (distance: 0.633)
> [3] sage_street.txt — Overview                             (distance: 0.653)
> ```
>
> All retrieved distances are above 0.6 — well above the 0.5 on-topic threshold — confirming that the corpus contains no relevant information. The model correctly declined to answer.

---

## Query Interface

The system is served through a Gradio web app ([app.py](app.py)) accessible at `http://localhost:7860`.

**Input fields:**
- **Your question** — a free-text box where the student types any food-related query about UC Davis dining.

**Output fields:**
- **Answer** — the LLM-generated response, grounded strictly to the retrieved context. If no relevant chunks were found, this box displays the fallback: *"I don't have enough information on that."*
- **Sources** — a programmatically generated list showing the source file, section name, and cosine distance for each of the top-5 retrieved chunks. This box is populated from ChromaDB metadata, not from the LLM.

**Sample interaction transcript:**

```
[Input]
Your question: What are the hours for Segundo Dining Commons on a Tuesday?

[Output — Answer]
The hours for Segundo Dining Commons on a Tuesday are 7 AM–10 PM.

[Output — Sources]
[1] ucd_segundo_dc.txt — Academic Year Hours  (distance: 0.170)
[2] ucd_segundo_dc.txt — Lunch                (distance: 0.222)
[3] ucd_segundo_dc.txt — Lunch                (distance: 0.222)
[4] ucd_segundo_dc.txt — Lunch                (distance: 0.223)
[5] ucd_segundo_dc.txt — Lunch                (distance: 0.223)
```

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What are the hours for Segundo Dining Commons on a Tuesday? | 7:00 AM – 10:00 PM | "The hours for Segundo Dining Commons on a Tuesday are 7 AM–10 PM." | Relevant | Accurate |
| 2 | Does The Gunrock accept meal swipes or AggieCash? | The Gunrock does not accept meal swipes or recharge billing. | "I don't have enough information on that." | Partially relevant | Inaccurate |
| 3 | Where can I get good garlic knots and a spinach stromboli near campus? | Yelp/Reddit chunks pointing to a local Italian restaurant or pizza spot | "I don't have enough information on that." | Off-target | Inaccurate |
| 4 | Which spots in Davis have the best spicy mango habanero wings? | Quora/Reddit chunks mentioning Wingstop or a local wing joint | "I don't have enough information on that." | Partially relevant | Inaccurate |
| 5 | Are there any places near the dorms that serve Indian street food like pav bhaji? | Latitude dining info regarding their Indian platform, or Reddit/Quora chunks mentioning local international food options | "I don't have enough information on that." | Partially relevant | Inaccurate |

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
"Does The Gunrock accept meal swipes or AggieCash?"

**What the system returned:**
"I don't have enough information on that." — The grounding instruction fired correctly because none of the 5 retrieved chunks contained The Gunrock's payment policy. The top 5 results were: `gunrock.txt | Reservations` (dist=0.436), `gunrock.txt | View More Special Meals & Events` (dist=0.440), `ucd_segundo_dc.txt | Aggie Swipe Plus Meal Plans` (dist=0.453), `gunrock.txt | We're Hungry for Feedback` (dist=0.453), and `gunrock.txt | The Gunrock` (dist=0.492). Every distance was above 0.43 — none cleared the on-topic threshold — and the payment policy chunk was never retrieved at all.

**Root cause (tied to a specific pipeline stage):**
The failure originates in the **chunking stage (Track A)**. The Gunrock page was split strictly by Markdown headers, which means the payment/billing information was isolated in its own thin chunk — likely just a line or two stating that meal swipes and recharge billing are not accepted. That chunk contained almost no semantic signal beyond the bare negative fact. When the query `"Does The Gunrock accept meal swipes or AggieCash?"` was embedded, the terms "AggieCash" and "meal swipes" had higher cosine similarity to the `ucd_segundo_dc.txt | Aggie Swipe Plus Meal Plans` chunk — a dense section that repeatedly uses both terms in a positive, descriptive context — than to the sparse Gunrock payment policy chunk that only uses them once in a negation. The embedding model treats "AggieCash" as a domain keyword and ranks whichever chunk uses it most richly, regardless of which restaurant the question is asking about.

**What you would change to fix it:**
Two targeted fixes. First, increase the entity-name injection in the embedded text for Track A chunks: instead of prepending only `"The Gunrock — "` to the chunk body, also append the restaurant name to the end, so sparse chunks carry entity weight on both sides for the encoder. Second, add a metadata pre-filter to ChromaDB retrieval: if the query contains a named entity that matches a known `source_file` (e.g. "Gunrock" → `gunrock.txt`), restrict the initial candidate pool to that file before re-ranking by distance. This prevents high-term-frequency chunks from unrelated restaurants from displacing on-topic but term-sparse chunks.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
The two-track chunking strategy defined in the spec — Track A for official UCD dining pages, Track B for social/unstructured sources — gave the implementation a clear routing rule before any code was written. Because the spec pre-committed to which sources belonged in each track and why (structured pages need semantic units like hours/menu/payment kept together; conversational reviews need sliding windows to prevent recommendation bleed), there were no architectural decisions left to make during coding. The `ingest_pipeline.py` implementation became a direct translation of the spec's reasoning into code rather than an open-ended design problem. The Evaluation Plan's 5 specific questions with expected answers was also directly useful: it gave a concrete, testable definition of "working correctly" at each milestone rather than leaving success ambiguous.

**One way your implementation diverged from the spec, and why:**
The spec called for "Markdown Element-Based Chunking" for Track A, splitting on headers like `### Hours`. When the scraped `.txt` files were examined, they contained no markdown at all — section labels like `Hours`, `Menu`, and `Contact Us` appeared as bare plain-text lines with no `#` prefix. A literal markdown splitter would have emitted one giant chunk per file, defeating the entire purpose. The implementation pivoted to a `_looks_like_header()` heuristic in [ingest_pipeline.py](ingest_pipeline.py) that detects bare capitalized label lines as structural headers using rules about line length, punctuation, and known section names. The spec's intent — keeping hours, menu items, and payment facts bundled as individual semantic units — was fully preserved, but the underlying mechanism is a plain-text heuristic rather than a markdown parser.

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

- *What I gave the AI:* I provided the Chunking Strategy section from planning.md along with the 10 pre-scraped `.txt` files in the `documents/` directory. The spec already contained my key design decisions: Track A documents (official UCD dining pages) should use structural header chunking rather than fixed-size splitting, because fixed chunks would break semantic units like hours, menu items, and payment info across chunk boundaries. Track B documents (Reddit, Yelp, Quora, TripAdvisor) should use a 500–800 character sliding window with 100-character overlap, with the source platform prepended to each chunk body for context.
- *What it produced:* `ingest_pipeline.py` with a `chunk_track_a()` function that split on detected section headers and a `chunk_track_b()` function implementing the sliding window. It also produced the `Chunk` dataclass and the `build_corpus()` orchestration function.
- *What I changed or overrode:* The spec called for "Markdown Element-Based Chunking" splitting on headers like `### Hours`, but the actual scraped `.txt` files contained no markdown at all — section labels appeared as bare plain-text lines. I directed the AI to replace the literal markdown parser with a `_looks_like_header()` heuristic that detects bare capitalized label lines as structural boundaries using rules about line length, punctuation, and a known-headers list. The chunking intent from my spec was preserved; only the detection mechanism changed to match the real data.

**Instance 2**

- *What I gave the AI:* I provided the Retrieval Approach section from planning.md (specifying `all-MiniLM-L6-v2`, 384-dimensional vectors, ChromaDB with cosine distance, top-k = 5) alongside the `ingest_pipeline.py` output from Instance 1. I also provided the Evaluation Plan's 5 test queries and the distance threshold (< 0.5) I had specified for on-topic results.
- *What it produced:* `embed_retrieve.py` with `build_index()` to embed and persist all chunks into a local ChromaDB collection, and `retrieve_context(query, k=5)` returning ranked results with text, metadata, and cosine distance. It also generated the `run_evaluation()` verification block that runs all 5 eval queries and prints distance scores.
- *What I changed or overrode:* The spec said to inject global page context (restaurant name) into chunk metadata only, not the body. After seeing retrieval fail to match sparse Track A chunks against entity-specific queries (e.g. a bare `Academic Year Hours` chunk never matching "Segundo hours"), I directed the AI to add a `_embed_text()` function that prepends the restaurant name to Track A chunk text at embed time — so the vector captures the entity — while keeping the stored document and metadata unchanged. This was a targeted fix to the entity-resolution gap I had flagged in the Anticipated Challenges section of planning.md.
