# Chunking review notes — Coke 10-K (Exercise C, Unit 2.2)

**Document:** Coca-Cola 10-K excerpt (`data/coke_10k_excerpt.txt`), ~60,754 characters
**Chunker:** `RecursiveCharacterTextSplitter` (structure-aware), via `chunk_document()`

## Findings

- At `chunk_size=500`, roughly 44% of chunk boundaries were flagged as
  potentially bad (starts lowercase / doesn't end in sentence-ending
  punctuation) — an estimated ~68 real bad boundaries out of ~155 chunks
  (each bad boundary flags both the chunk before and after it, so raw
  flagged-chunk count overstates the true number by roughly 2x).
- Root cause: long legal/regulatory sentences (e.g. the Forward-Looking
  Statements section) exceed 500 characters on their own. Structure-aware
  chunking's priority order (headers → paragraphs → sentences → fixed-size
  fallback) has no higher-priority option left in this case, so it falls
  back to a raw character cut _within_ the oversized sentence — a
  documented, expected limitation, not a bug.
- Raising `chunk_size` to 1000 reduced flagged chunks by roughly 3x
  (136 → 44), confirming the sentence-length theory.
- **Not every flagged chunk is a chunking problem.** Some are PDF-extraction
  artifacts baked into the source text itself — e.g. missing whitespace
  after periods (`"...emerging beverages.We own and market..."`). The
  chunker faithfully preserves this mess; it doesn't create it. This
  needs a text-cleaning step _before_ chunking, not a chunking fix.

## Decision for Module 2 Portfolio Project

- Default to `chunk_size=1000` (not 500) for dense financial/legal
  documents — meaningfully fewer bad boundaries, still small enough for
  focused retrieval.
- Add a pre-processing cleanup pass before chunking:
  - Strip repeated footer/timestamp/URL junk (seen in the raw PDF extract)
  - Normalize missing whitespace after sentence-ending punctuation
- Metadata (`source`, `section`, `chunk_index`) attachment pattern from
  Unit 2.2 carries forward as-is — no changes needed there.
- Overlap's effect was marginal on this document specifically, because
  structure-aware chunking already avoids most bad boundaries when
  sentences fit within `chunk_size` — overlap matters much more as a
  safety net for fixed-size chunking, which has no boundary-awareness at
  all. Still worth keeping a moderate overlap (e.g. 100–150 chars) as
  cheap insurance for the cases that do fall back to fixed-size cuts.
