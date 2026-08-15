.PHONY: bench bench-plot test lint

VENV ?= .venv/bin
RESULTS ?= $(shell ls -t bench/results-*.json 2>/dev/null | head -1)

bench:            ## Run the coding-model benchmark -> bench/results-<ts>.json
	$(VENV)/python -m guru.bench

bench-plot:       ## Plot the latest results (override with RESULTS=...)
	@test -n "$(RESULTS)" || { echo "no results file; run 'make bench'"; exit 1; }
	$(VENV)/python -m guru.bench_plot $(RESULTS)

test:             ## Run the test suite
	$(VENV)/python -m pytest -q

lint:             ## Lint with flake8
	$(VENV)/flake8 guru bench tests
