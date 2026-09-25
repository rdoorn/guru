.PHONY: bench bench-plot test test-sandbox lint typecheck yarn ledger-report eval-fast

VENV ?= .venv/bin
RESULTS ?= $(shell ls -t bench/results-*.json 2>/dev/null | head -1)

# YaRN long-context build. Ollama cannot enable YaRN on an existing model
# (its Modelfile rejects rope-scaling params), so we pull a GGUF that already
# has YaRN baked in; guru then auto-detects the extended (128K) ceiling.
# Override to extend other models: make yarn YARN_REPO=... YARN_QUANT=...
YARN_REPO  ?= hf.co/unsloth/Qwen3-14B-128K-GGUF
YARN_QUANT ?= Q4_K_M

bench:            ## Run the coding-model benchmark -> bench/results-<ts>.json
	$(VENV)/python -m guru.bench

bench-plot:       ## Plot the latest results (override with RESULTS=...)
	@test -n "$(RESULTS)" || { echo "no results file; run 'make bench'"; exit 1; }
	$(VENV)/python -m guru.bench_plot $(RESULTS)

ledger-report:    ## Judge vs heuristic vs labels per decision point (guru.ledger_cli report)
	$(VENV)/python -m guru.ledger_cli report

eval-fast:        ## Fast eval gate x3: --tags fast --repeat 3; extra flags via EVAL_ARGS='--routing ... --allow-spend'
	$(VENV)/python -m guru.evals run --tags fast --repeat 3 $(EVAL_ARGS)

test:             ## Run the test suite (container tests excluded)
	$(VENV)/python -m pytest -q -m "not sandbox"

test-sandbox:     ## Run the container integration tests (needs Colima)
	$(VENV)/python -m pytest -q -m sandbox tests/test_sandbox_integration.py

lint:             ## Lint with flake8
	$(VENV)/flake8 guru bench tests evals

typecheck:        ## Type-check with mypy (local; there is no CI)
	$(VENV)/python -m mypy guru

yarn:             ## Pull a YaRN-baked (128K) build: make yarn [YARN_REPO=.. YARN_QUANT=..]
	ollama pull $(YARN_REPO):$(YARN_QUANT)
