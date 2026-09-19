# VetScore: Risk-Weighted Fact Verification for Veterinary Long-Form QA with Citations

Code and data for the paper VetScore: Risk-Weighted Fact Verification for Veterinary Long-Form QA with Citations.

## Dataset

The dataset consists of three files, each with 1,200 records (use the `id` field to join the files):

| File | Contents |
|---|---|
| `segments.jsonl` | Inputs: query, segment text, decomposed claims, cited excerpts |
| `annotations_scoring.jsonl` | Human annotations for the harm potential scoring: per-rater and aggregate scores |
| `annotations_verification.jsonl`| Human annotations for the verification task: per-rater verification scores |

Since every segment was annotated for both tasks, the two annotation files both cover the same 1,200 segments.

Every citation in this dataset quotes an excerpt from a veterinary article on PubMed Central. Those passages are third-party copyrighted text, so what we can publish depends on each article's licence. Each excerpt's `license` field records the terms that apply.

### `segments.jsonl`

The file contains text segments sampled from the generated outputs, each with the following fields:
* `id`: unique identifier of the segment
* `query`: query used to generate the output
* `topic`: query translated to a topic formulation (see Section 4.3 in the paper)
* `model`: model used to generate the output
* `segment_index`: index of the segment in the corresponding output
* `text`: segment text
* `claims`: individual claims contained in the segment (as decomposed by Gemini 3 Flash) 
* `excerpts`: list of excerpts provided by the generating model for the given segment

Each excerpt contains these fields:
* `source_id`: unique identifier of the source, used internally
* `source_url`: the URL from which the source was retrieved
* `source_title`
* `source_authors`
* `source_journal`
* `source_year`
* `license`: license of the source
* `match`: the type of match of the given excerpt (see Section 4.2 in the paper)
* `verbatim`: whether the excerpt is found verbatim in the source
* `excerpt`: the excerpt text as provided by the output generating model (`null` if we were not able to include it due to licensing reasons)

The `excerpt` field contains the text for 1,154 of the 1,470 excerpts. The other 316 excerpts are blanked, for one of four reasons, each identifiable from `license` and `verbatim`:

| Reason | Excerpts | How to recognize it |
|---|---:|---|
| No reuse licence, outside the PMC OA subset | 226 | license is `No open licence (not in PMC OA subset)` |
| No CC licence, inside the PMC OA subset | 48 | license is `No CC licence (in PMC OA subset)` |
| Not the article's own words, and the source licence either forbids (ND) or conditions (SA) the sharing of adapted material | 41 | license is a CC licence, `verbatim` is `false` |
| The model cited a source id that does not exist | 1 | `license` is `null` |

The dataset is therefore **not fully reproducible as published**. Please contact the authors if you need the full dataset for reproducibility.

### `annotations_scoring.jsonl`

Human annotations for the harm potential scoring task. Each record contains the following fields:
* `id`: segment ID
* `raters`: anonymised codes of annotators who rated the segment's claims
* `claims`: claim scores in the same order as the segment's `claims` in `segments.jsonl` (entry *i* holds the ratings for claim *i*)

Each record in the `claims` field consists of the following fields:
* `scores`: raw scores for the claim, ordered by `raters`
* `mean`: simple average of the raw annotated scores on that claim 

### `annotations_verification.jsonl`

Human annotations for the verification task. Same structure as `annotations_scoring.jsonl`, with two additional fields for each claim:

* `pcm_theta`: latent harm potential of the claim as obtained from the fitted partial credit model (PCM)
* `pcm_fair`: aggregate annotator-adjusted score used in our experiments

## VetScore

The `vetscore` package implements the VetScore pipeline.

### Install

Requires Python 3.10 or newer.

```bash
pip install -e .
pip install -e '.[ollama]'    # also enables backend: ollama
```

Ollama is an extra because its SDK is only imported when you select that
backend. OpenRouter is used by default.

### Usage

A run is specified in a YAML config file:

```bash
export OPENROUTER_API_KEY=...
vetscore configs/gemma-4-31b.yaml
```

Alternatively, set `OPENROUTER_API_KEY` in `.env` (see `.env.example`).

To run against a local [Ollama](https://ollama.com) server, install the
`ollama` extra. The server defaults to `OLLAMA_HOST` or `http://localhost:11434`.

```yaml
input: data/segments.jsonl
output: results/gemma-4-31b.json
backend: ollama
model: gemma4:31b-it-bf16
reasoning_level: none
skip_segmentation: true
skip_decomposition: true
```

## Licensing

The code and the dataset are licensed separately. See [`LICENSE`](LICENSE).

**The code:** everything under `vetscore/` and `configs/` MIT licensed.

**The dataset:** everything under `data/` (not under a single licence).
