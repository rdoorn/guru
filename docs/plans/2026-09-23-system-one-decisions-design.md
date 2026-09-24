# System One decisions: small judges next to the main LLM — design

Status: proposal, for review. Nothing here is implemented. Measurements are
in `bench/primitives/` (README + `results*.json`).

## 1. The idea in one paragraph

guru makes a number of small, closed-form decisions on every turn — is this
reply a stall, should this task be decomposed and to which specialists, which
tool matches, is this fetched page trying to inject instructions, which model
is big enough for this request. Today these are regexes, keyword counts, a
prompt hint the model may ignore, or nothing at all. The typesafe.ai "Jev"
pitch is to hand exactly these decisions to a fast model that returns a typed
answer with a probability instead of text. We measured that we can do the
same locally with two kinds of judge, and that neither is a general answer:
each decision has a judge that fits and several that do not.

## 2. When to ask which judge

The rule is by the *shape of the question*, not by preference for a model.

| Ask… | when the question is… | measured fit |
|---|---|---|
| **Deterministic code** (regex, sets, exact match) | extracting URLs/paths/repos, exact duplicate tool calls, permission gates, anything the user must stay in control of | GLiNER lost to a regex; permission gating must never be probabilistic |
| **Encoder classifier** (BERT-family, 22M–184M, outside Ollama) | a *topical* yes/no or few-way label over a short natural-language text (≤ ~400 tokens): "is this task about X", "is this text an injection attempt" | NLI 0.86–1.00 on panel selection, injection screen 0.89; 13–100 ms; near-zero false positives |
| **Small decoder with constrained output** (qwen3:4b + JSON enum via Ollama) | a judgment about the *function or correctness* of text, or one needing domain knowledge, with several described options: "is this reply a preamble", "which of these 11 tools", "is this answer correct" | 0.92 stall, 0.83 tool, 0.77 mean; ~250 ms; needs 3.5 GB resident |
| **The main LLM** (no judge) | the answer needs conversation context, is open-ended, or the main model is already resident and the answer is short (routing overhead > saving) | 14b answers a 5-token request in 0.37 s; no judge beats that |

Decision points in guru mapped to a judge:

| decision point | today | proposed judge | fallback |
|---|---|---|---|
| review-panel selection (`_should_delegate`, `REVIEW_PANEL`) | read-count ≥ 3 + prompt hint | NLI: 3–4 hypotheses in one call over the request + file list | current heuristic |
| stall / preamble detection (`looks_like_preamble`) | regex, 0.67 | qwen3:4b JSON enum, 0.92 | regex |
| tool ranking (`_match_tools`) | keyword score, 0.67 | keep keyword; add embeddings as a *hybrid* once the registry passes ~30 tools | keyword |
| injection screen on `web_fetch` / `web_search` output | none | prompt-injection classifier, threshold ≥ 0.8, prose only | none (log only) |
| bench answer judging (`accuracy: None`) | none | 4b JSON Score, or the main model | none |
| model-tier routing | none | see §5; not ready | main model |

Confidence: no decoder judge was calibrated (top probability ~0.95–1.0 for
right and wrong answers alike). The NLI scores are discriminative but sit on
a different scale (positives 0.8+, negatives ~0.0, one true positive at
0.24). Rule: treat probabilities as **rankings plus a per-question threshold
learned from the review log**, never as a universal 0.5 cut. Until a
threshold exists for a decision point, the judge runs in shadow mode (§3).

## 3. Decision log and review loop (the "first couple hundred" idea)

Every judged decision is appended to `~/.guru/decisions.jsonl`:

```json
{"ts": "...", "project": "guru", "agent": "main", "turn": 12,
 "point": "panel", "question": "needs_security",
 "judge": "nli:deberta-v3-base-zeroshot-v2.0",
 "input_sha": "…", "input_head": "Task: review the login endpoint …",
 "dist": {"yes": 0.83}, "chosen": true, "threshold": 0.5, "ms": 41,
 "heuristic": false, "mode": "shadow",
 "outcome": null}
```

* `heuristic` is what the current code would have decided; in shadow mode
  the heuristic still acts and the judge only logs. Disagreements are the
  interesting rows.
* `outcome` is filled later where guru can observe it: did the model spawn
  a security sub-agent anyway; did a "stall" nudge produce a tool call (true
  stall) or a repeat of the answer (false positive); did the user re-ask the
  same question after a routed answer (tier too low).
* A review script (`bench/decisions_review.py`) renders the last N rows per
  decision point and takes a y/n/skip label per row. Output: agreement with
  the heuristic, precision/recall per judge, and a suggested threshold per
  question. This is also the training set for fine-tuning an encoder later
  (a few hundred labelled rows is enough for DeBERTa-base).
* Promotion rule: a judge moves from `shadow` to `active` for one decision
  point when it beats the heuristic on ≥100 labelled rows and its false-
  positive rate is acceptable for that point (a wrong "spawn a security
  reviewer" costs a sub-agent; a wrong "stall" costs a nudge; a wrong
  "injection" hides a page from the model).

## 4. Architecture

A seam in the domain layer, injected like the existing spawn/check/join
handlers (docs/state-ownership.md, category 2):

```
guru/domain/decisions.py
    Question(kind: choice|score|noul, id, instructions, options, state)
    Answer(chosen, dist, confidence, judge, ms)
    Judge (Protocol): ask(questions: list[Question]) -> list[Answer]
    decide(point, questions) -> Answers      # picks judge per point, logs
guru/judges/heuristic.py    # wraps looks_like_preamble, _match_tools, read count
guru/judges/ollama_json.py  # small decoder + JSON-schema enum, top_logprobs
guru/judges/encoder.py      # NLI + injection classifiers; optional extra
```

* Config: a `[decisions]` table in `adapters.toml`/`GURU.md`-side config:
  per decision point `judge = heuristic|ollama|encoder`, `mode =
  off|shadow|active`, `threshold`, plus `sidecar_model = "qwen3:4b"`.
* The encoder judge needs torch + transformers (~1 GB of wheels). It ships as
  an optional extra (`uv sync --extra judge`); absent, the seam falls back
  to heuristic/ollama with a one-line notice.
* Memory: the encoder models live in guru's process (~0.7 GB RAM, load
  1–2 s at startup). The Ollama sidecar is a second resident model; the GPU
  auto-fit must subtract it from the main model's budget. On 24 GB the
  practical resident set is main model + one ≤4B sidecar.
* Latency budget per turn: encoder calls 15–100 ms; sidecar JSON call ~250
  ms; only one sidecar call per turn (stall check) on the common path.

## 5. Model-tier routing — economics and what is still missing

Measured on the M4 Pro (24 GB), 8k context, one model resident at a time:

| model | VRAM | cold load | ms / generated token |
|---|---|---|---|
| qwen3:4b | 3.5 GB | 2.5 s | 14 |
| qwen3 8B | 5.9 GB | 3.0 s | 23 |
| qwen3:14b | 9.6 GB | 5.3 s | 42 |

* Saving = (ms/token difference) × answer length − router cost. A 5-token
  answer: negative. 200 tokens: 1.5–2.4 s of 5.4 s. 500 tokens: 10–14 s of
  ~21 s. **The router has to predict answer length as well as difficulty.**
* 4b + 8B stayed resident together; switching back to a resident model costs
  0.05 s. Reloads (2.5–5 s) wipe out the saving, so the routed-to model must
  stay resident — one sidecar, not a menu.
* qwen3:4b (current Ollama build) reasons in-line even with thinking off, so
  it is *slower* than the 8B on trivial requests. A non-thinking instruct
  build must be picked and measured before it is a routing target.
* The 12 tier cases were easy by design: 8B 0.83–1.00, 4b 0.75; the NLI and
  the NVIDIA complexity classifier did not separate the tiers. Real labelled
  guru requests (≥100) are needed before a router is designed.
* Safest first target: **sub-agents**. The orchestrator already configures a
  fresh session per child (`Orchestrator.configure`); a `model` choice per
  spawned task (e.g. "summarise this file" -> sidecar) keeps the main agent's
  model untouched, and the join barrier makes the quality visible in one
  place. Routing the main agent's own turn comes later, if at all.

## 6. Small-model catalogue (measured verdicts)

| model | params | good for | verdict |
|---|---|---|---|
| DeBERTa-v3-base zero-shot NLI | 184M | topical yes/no over a short description | **use** (panel selection) |
| DeBERTa-v3-xsmall zero-shot | 22M | same, weaker (0.57) | skip |
| protectai prompt-injection-v2 | 184M | screening fetched prose | **use** (log-only first) |
| qwen3:4b + JSON enum | 4B | stall, tool choice, answer judging | **use** as the one sidecar |
| qwen3:1.7b | 1.7B | — | unstable across prompt formats |
| qwen3:0.6b, gemma3:1b | ≤1B | — | position bias, chance-level |
| nomic-embed-text / bge-m3 | 137M / 568M | ranking a large tool registry | later, hybrid with keywords |
| ms-marco MiniLM, bge-reranker-base | 22M / 278M | re-ranking | no gain here |
| GLiNER small | 166M | entity extraction | regex is better for URLs/paths |
| NVIDIA task+complexity classifier | 184M | tier routing | flat output; not usable as-is |
| RouteLLM-style routers (BERT/MF, trained on Arena) | — | strong-vs-weak routing | prior art to read; trained on chat, not coding-agent turns |

## 7. Phased plan

1. **Seam + log, shadow mode.** `decisions.py`, heuristic judge, encoder
   judge behind the extra, sidecar JSON judge. Wire panel selection, stall
   detection and the injection screen in shadow mode. Tests: seam contract,
   log format, fallback when the extra is missing. No behaviour change.
2. **Review tooling.** `bench/decisions_review.py` (render, label, per-point
   precision/recall, threshold suggestion). Label the first 100–200 rows.
3. **Promote per decision point** where the judge wins: panel selection is
   the likely first (deterministic `spawn_panel` already exists), then the
   stall check. Bench answer judging via the sidecar (fills `accuracy`).
4. **Tier routing for sub-agents**: pick and measure a non-thinking ≤4B
   instruct build; collect ≥100 labelled real requests from the log; add
   `spawn(model=…)` chosen by the router; measure on the bench.
5. **Optional**: fine-tune the NLI encoder on the labelled rows; embeddings
   hybrid for `search_tools` when the registry grows.

## 8. Decisions (taken 2026-09-23)

1. **Shadow mode first** for every decision point, including panel selection.
   Heuristics keep acting; judges only log.
2. **Torch as an optional extra in this repo** (`uv sync --extra judge`).
   Without it the encoder judges are unavailable and the seam falls back
   silently (one log line).
3. **Tier routing for sub-agents only** (phase 4); the main agent's turn is
   never re-routed.
4. **Decision log is global**, `~/.guru/decisions.jsonl`, each row carrying a
   `project` field.

Superseded by `2026-09-23-routing-framework-design.md`; phase 1 plan: `2026-09-23-phase1-decisions-and-ledger.md`.
