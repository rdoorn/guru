# Primitives probe — can a small local model be the "System One" judge?

Experiment (2026-09-23) testing whether a small local model can answer typed
questions (Choice / Score / Noul, in the sense of typesafe.ai's Jev) fast
enough and accurately enough to steer guru: pick a model tier for a request,
choose a tool, select a review panel, detect a stalled reply, judge an answer.

Two mechanisms were measured on the same 69 labelled cases (`cases.py`):

* `probe.py` — decoder LLMs via Ollama, reading next-token log-probabilities
  over the option letters (no text generation). Variants: `plain` (prompt +
  "Answer:"), `plain-rev` (options reversed, to expose position bias), `chat`
  (system prompt "reply with one letter"), `json` (JSON-schema-constrained
  enum, i.e. grammar-forced).
* `probe_nli.py` — zero-shot NLI cross-encoders (DeBERTa-v3) via transformers,
  run from a throwaway venv (torch is not a guru dependency).

Results are in `results*.json`; `report.py` renders the LLM tables.

## Headline numbers (mean accuracy over the 7 tasks, best variant)

| judge | params | best variant | mean acc | typical latency |
|---|---|---|---|---|
| qwen3:0.6b | 0.6B | json | 0.46 | 15–90 ms |
| gemma3:1b | 1B | plain | 0.39 | 50–220 ms |
| qwen3:1.7b | 1.7B | chat | 0.73 (0.63 other variants) | 50–370 ms |
| qwen3:4b | 4B | json only (free-form emits "First, …") | 0.77 | 230–340 ms |
| qwen3 8B (abliterated) | 8B | plain-rev / chat | 0.76–0.78 | 200 ms short, 1.7 s for the 600-token tool prompt |
| DeBERTa-v3-xsmall zeroshot | 22M | n/a | 0.57 | 14–50 ms |
| DeBERTa-v3-base zeroshot | 184M | n/a | 0.70 | 13–50 ms (160 ms for 11-way tool choice) |
| guru heuristics | – | regex / keyword | stall 0.67, tool 0.67 | ~0 |

## Findings

1. Below ~1.7B the decoder models are not judges. qwen3:0.6b and gemma3:1b
   answer "A" regardless of content (position bias): reversing the option
   order flips their answers. Accuracy is at or below chance.
2. 1.7B works on some questions (security/architecture panel: 1.00 in `chat`)
   but is unstable across prompt formats (tier: 0.33 in `chat`, 1.00 in
   `json`). Not robust enough to gate behaviour on without per-task tuning.
3. Constrained decoding is required for robustness. qwen3:4b starts every
   free-form reply with "First, the…" even with thinking disabled, so the
   single-token trick gets zero mass on the letters. With a JSON-schema enum
   it becomes the best small decoder (stall 0.92, tool 0.83, tier 0.75,
   security panel 1.00) at ~250 ms. Cost of the constraint: ~2–3x latency
   (6–8 decoded tokens instead of 1).
4. The 8B is the most accurate and stable, but its prefill on this machine
   makes it slower than a generative small model would be for short answers,
   and it over-flags: it said "yes" to nearly every "review …" task on all
   three panel questions. Aggregate accuracy hides that; per-case inspection
   does not.
5. Confidence is not calibrated on any decoder model: mean max-probability
   is ~0.95–1.0 for both correct and wrong answers. Do not use the raw
   probability as a confidence gate; use it as a ranking only, or calibrate
   with held-out labels.
6. NLI cross-encoders are the real surprise for Noul. DeBERTa-v3-base (184M,
   13 ms) matched or beat the 8B on the panel questions and was genuinely
   discriminative (near 0 on negatives, >0.8 on positives). It missed two
   cases that need domain knowledge ("writes user-provided paths to disk" as
   security, "Helm chart + alerting" as SRE at 0.24), and it is weak on
   multi-way Choice with long option descriptions (tool 0.42) and on
   judging answer correctness (0.60). Its scores are low overall, so the
   0.5 threshold needs tuning per question.
7. Model-tier routing (the "use a small model when it suffices" idea) is
   learnable: 8B 0.83–1.00, 4B-json 0.75, NLI-base 0.50. The 12 cases are
   easy by design; a real evaluation needs a labelled sample of actual guru
   requests.
8. Memory: the sidecar costs 0.85–1.8 GB VRAM for 1–1.7B, ~2.6 GB for 4B,
   5.4 GB for 8B, on top of the main model. Cold load was 16 s for the
   1B/1.7B models when another model had to be evicted; keep the judge
   resident (`keep_alive`) or the latency win disappears.

## Recommendation

* Noul-style gates (needs security reviewer? is this reply a stall?): a
  184M NLI cross-encoder is the right tool — 13 ms, CPU-capable, no Ollama
  round trip. It needs torch, so it belongs behind an optional extra.
* Choice / Score with several options (tool selection, tier routing, answer
  judging): qwen3:4b with a JSON-schema enum via Ollama (~250 ms), or the
  main 8B+ model if it is already resident. Embeddings remain the right
  answer for ranking a 200-tool registry.
* Do not gate on probabilities until calibrated; treat them as rankings.
* Next step if pursued: label ~100 real guru requests for tier routing and
  re-run both probes on them before designing the seam.

## Reproduce

```bash
ollama pull qwen3:0.6b qwen3:1.7b gemma3:1b qwen3:4b
.venv/bin/python -m bench.primitives.probe --out bench/primitives/results.json
.venv/bin/python bench/primitives/report.py
uv venv /tmp/nli-venv && uv pip install --python /tmp/nli-venv/bin/python \
    torch transformers sentencepiece protobuf requests beautifulsoup4 ddgs \
    rich prompt-toolkit ollama
PYTHONPATH=. /tmp/nli-venv/bin/python bench/primitives/probe_nli.py --device mps
```

## Addendum (same day): switching economics and more small models

### Routing to a smaller model — what it actually saves (M4 Pro, 24 GB, 8k ctx)

`switch_cost.py`, one model resident at a time, `think=False`:

| model | VRAM | cold load | trivial request (wall) | medium request (~150–200 tok) | gen tok/s | ms per generated token |
|---|---|---|---|---|---|---|
| qwen3:4b | 3.5 GB | 2.5 s | 0.61 s (rambled 32 tok) | 2.96 s | 62–70 | 14 |
| qwen3 8B (abliterated) | 5.9 GB | 3.0 s | 0.21 s (5 tok) | 3.89 s | 43–52 | 23 |
| qwen3:14b | 9.6 GB | 5.3 s | 0.37 s (5 tok) | 5.40 s | 24–29 | 42 |

Coexistence: 4b loaded next to a resident 8B in 2.4 s; both stayed resident
(9.6 GB total); switching back to the 8B cost 0.05 s. 14b + 4b (~13 GB) is
plausible on 24 GB but was not tested after the 27B incident; 14b + 8B is not.

What this means:

* The saving is per generated token: 14b -> 8B saves ~19 ms/token, 14b -> 4b
  ~28 ms/token. A 5-token answer saves nothing (the router alone costs
  15–250 ms). A 200-token answer saves 1.5–2.4 s of 5.4 s. A 500-token answer
  saves 10–14 s of ~21 s. Routing pays off only when the answer is long AND
  a smaller model is good enough, so the router must predict both.
* qwen3:4b (current Ollama build) writes its reasoning in-line ("Okay, so I
  need to…") even with thinking off, so for trivial requests it is SLOWER
  than the 8B. As a generation target it needs a non-thinking instruct build
  (to be tested); as a JSON-constrained judge it is fine.
* Cold loads are 2.5–5 s once; the win depends on keeping the small model
  resident alongside the main one. On 24 GB that means main model + one
  sidecar of at most ~4B, plus encoder classifiers (<1 GB, outside Ollama).

### More small models, each against one guru use case (`probe_models.py`)

| model | params | use case tested | result | latency | verdict |
|---|---|---|---|---|---|
| DeBERTa-v3-base zero-shot NLI | 184M | panel selection (topical yes/no) | 0.86–1.00, discriminative | 13–100 ms | use, with tuned wording + threshold |
| protectai deberta-v3 prompt-injection-v2 | 184M | screen web_fetch / tool output | 0.89 (8/9); one false positive on raw code at 0.59 | 94 ms | use on web content, threshold ≥0.8 |
| nomic-embed-text (Ollama) | 137M | tool ranking, top-1 of 11 | 0.67 (= keyword baseline) | 11 ms/query | not better at 11 tools; the scaling path for 200; try hybrid with keywords |
| bge-m3 (Ollama) | 568M | tool ranking | 0.67 | 18 ms/query | same as above, heavier |
| ms-marco MiniLM-L-6 reranker | 22M | tool ranking | 0.67 | 54 ms | no |
| bge-reranker-base | 278M | tool ranking | 0.50 | 74 ms | no |
| GLiNER small v2.1 | 166M | extract URL / path / repo from a request | split URLs, missed paths | 17 ms | no — a regex does this |
| nvidia prompt-task-and-complexity-classifier | 184M | tier routing | flat 0.27–0.35 across all tiers; "Open QA" for every request | 124 ms | not usable off the shelf |

Injection-screen detail: every injection (including an HTML-comment one and
a "AI reading this" one) scored 1.00; a bare Python snippet was a false
positive at 0.59, so score code and prose differently or skip code.
