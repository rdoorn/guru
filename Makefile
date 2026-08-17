.PHONY: bench bench-plot test lint typecheck yarn

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

test:             ## Run the test suite
	$(VENV)/python -m pytest -q

lint:             ## Lint with flake8
	$(VENV)/flake8 guru bench tests

typecheck:        ## Type-check with mypy (local; there is no CI)
	$(VENV)/python -m mypy guru

yarn:             ## Pull a YaRN-baked (128K) build: make yarn [YARN_REPO=.. YARN_QUANT=..]
	ollama pull $(YARN_REPO):$(YARN_QUANT)
