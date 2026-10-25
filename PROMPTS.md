# Prompt Version History

Locus uses LLM/VLM components in two places. This file tracks every prompt version,
what changed, and the evaluation evidence that justified the change.

---

## A — CLIP Category Prompts (`visual_engine/clip_labels.py`)

Two separate prompt sets are maintained:

- **`CLIP_PROMPTS`** — index-time embeddings. Context: product-on-white background images.
- **`QUERY_CLIP_PROMPTS`** — query-time embeddings. Context: user photos (person wearing, street, mirror).

Keeping them separate is a deliberate design decision (see Tradeoff 3 in `docs/tradeoffs.md`):
product images and real-world query photos occupy different regions of the CLIP embedding space,
so using the same prompt for both degrades retrieval accuracy.

### Version history

| Version | Prompt set affected | Change | Reason / Evidence |
|---------|--------------------|---------|--------------------|
| v1 | Both (single set) | Long descriptive prompts per category (sentence-length) | Initial baseline |
| v2 | `CLIP_PROMPTS` | Shortened to token sequences; added `"halterneck"` to tops | Long prompts pushed halterneck dresses into the dress cluster, causing tops→dress misclassification on halterneck tops. Recall@5 +2 pp. |
| v3 | Split | Created separate `QUERY_CLIP_PROMPTS` using `"a photo of a person wearing..."` framing | Product-on-white and person-wearing photos have different CLIP anchor points. Splitting improved query-side top/dress separation. |
| v4 | `CLIP_PROMPTS` | Added `"jacket coat"` to outerwear; `"shoes sandals heels"` to shoes | Regression: longline wool coat was classified as top; strappy sandal as skirt. Both fixed by extending the token sequence. |
| v5 (current) | `CLIP_PROMPTS` | Added `"tights"` to leggings prompt | Leggings and pants share large CLIP embedding overlap; `"tights"` widens the gap. Recall@5 on leggings queries +4 pp. |

**How to evaluate:** `python mlops/evaluate_recall.py --gateway-url http://localhost:8000 --k 5 --mlflow`
Results logged under MLflow experiment `locus_recall_eval`.

---

## B — Attribute Tagger Prompts (`attribute_tagger/tagger.py`)

Current version: **`TAGGER_PROMPT_VERSION = "v5"`** (line 28 of `tagger.py`).
This constant is included in every MLflow run so each experiment records which prompt was active.

The tagger extracts 7 structured fields per crop: `colors`, `style`, `silhouette`, `occasion`,
`pattern`, `material_feel`, `trend_tags`.

### Version history

| Version | Change | Reason / Evidence |
|---------|---------|-------------------|
| v1 | Single generic prompt, no vocabulary constraints | Category-agnostic baseline. Gemini returned free-text values (e.g. `"business casual"`, `"knitted"`) that did not match downstream filter vocabulary. |
| v2 | Added closed vocabulary lists for `style`, `pattern`, `material_feel` in the prompt | Reduced field rejection rate from ~35 % to ~12 %. Filter-compatible values increased. |
| v3 | Added 11 category-specific hint blocks (`_CATEGORY_HINTS` in `tagger.py`) | Shoes and accessories have different field semantics: for shoes, `silhouette` should describe heel type, not dress cut. Category hints cut cross-semantic errors in half on footwear queries. |
| v4 | Added server-side alias normalisation table (`_STYLE_ALIASES`, `_PATTERN_ALIASES`, `_MATERIAL_ALIASES`); added `_confidence` output field | Even with vocabulary in the prompt, Gemini occasionally returns close-but-not-exact values (e.g. `"boho"` instead of `"bohemian"`). Alias normalisation handles these without another LLM call. `_confidence` = fraction of constrained fields already in-vocab, tracked in Prometheus via `locus_tagger_failures_total`. |
| v5 (current) | Changed constrained fields to use "MUST be exactly one of" wording and exhaustive lists; tightened color guidance to require compound names ("navy blue" not "blue") | A/B experiment `tagger_prompt_tuning` (MLflow, 2026-05-07) across 35 golden queries: vocab_hit_rate 0.952→1.000 (+4.8 pp), completeness 0.935→0.951 (+1.6 pp), confidence 0.933→1.000 (+6.7 pp). Winning variant: `v5a_strict_vocab`. |

### Provider chain

The tagger always attempts providers in this order, never raising to the caller:

1. **OpenRouter (Gemini 2.0 Flash)** — primary. Custom rate limiter: 2 s minimum gap (~30 RPM). Exponential backoff on 429/504.
2. **Google Gemini REST** — fallback when OpenRouter returns 429 or is unavailable.
3. **`{}`** — graceful degradation. Search results are returned without attribute data; the UI omits the attributes panel.
