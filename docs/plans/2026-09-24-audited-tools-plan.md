# Audited coding toolset (no shell) — design and implementation plan

> **For Claude:** execute task-by-task with subagents; one commit at the end
> of each chunk after `make lint && make typecheck && make test` are green.

**Goal:** give guru's workers real coding and verification capability —
outline, symbol search, tests, syntax and lint checks, git diff, patches —
through fixed Python procedures with minimal digests, full audit, and no
shell. Decisions (2026-09-24): pytest and unittest only, `apply_patch`
included, process jailing (container) later.

**Principles**

1. No command strings. Every subprocess is a fixed argv list guru builds;
   the model chooses verbs and targets only.
2. Minimal feedback. Default digest ≤ ~600 chars; a `detail` argument expands
   one item; the full output goes to the ledger transcript, not the model.
3. Existing gates apply: allow-lists, access modes, secret redaction for
   remote adapters, per-call ledger rows. Path resolution rejects symlinks
   that leave the project.
4. Layers: domain (pure digests, AST, patch algebra, policy), repository
   (tool_events stream in the JSONL ledger, `.guru/tools.toml`), endpoint
   (subprocess runner, git, tool registry entries, `/tools` command).

---

## Chunk A — foundations

### A1 `guru/domain/procs.py` (endpoint-ish runner; stdlib only)
- `Limits(timeout_s=120, cpu_s=120, mem_mb=2048, fsize_mb=64, out_kb=256)`
  with defaults from `config.PROC_*` (settings `[tools.limits]`).
- `run(argv: list[str], cwd: Path, limits: Limits, env_extra: dict = {})
  -> ProcResult(argv, returncode, stdout, stderr, seconds, timed_out,
  truncated)`; environment built from scratch: `PATH`, `HOME` = a temp dir,
  `LANG`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=<cwd>`, plus `env_extra`;
  `preexec_fn` sets RLIMIT_CPU/AS/FSIZE (skip silently where unsupported);
  `subprocess.run(..., timeout=)`; stdout/stderr capped to `out_kb` with a
  `truncated` flag. Never invoked with `shell=True`. Refuses a cwd outside
  the allow-listed read dirs (`files.ensure_path_allowed`).
- Tests: argv fixed, env scrubbed (a secret in `os.environ` does not reach
  the child), timeout → `timed_out`, output cap → `truncated`, rlimit path
  at least exercised on darwin, cwd outside allow-list refused.

### A2 tool events (audit)
- `ledger.record_tool_event(tool, args, seconds, ok, produced_bytes,
  shown_bytes, files_touched: list[str], denied: str = '')` → stream
  `tool_events` (base row + agent/task/turn keys). Called from
  `tools.execute_tool` for every tool (existing ones too): `produced_bytes`
  = len(raw result), `shown_bytes` = len(result after digest/redaction).
- `/tools` slash command (cli + tui dispatch): last turn's events as a
  compact table (tool, args head, seconds, shown/produced, denied).
- `bench/ledger_report.py`: a "Tools" section (calls per tool, mean seconds,
  bytes shown vs produced, denials).

### A3 tool policy
- `.guru/tools.toml` (project) with `[tools] enabled = [...]`,
  `disabled = [...]`, `[tools.tests] runner = "pytest" | "unittest"`,
  `[tools.limits] timeout_s = ...`. Loaded by `repositories/settings.py`
  (`load_tools_policy`); `tools.execute_tool` refuses a disabled tool with a
  clear message and a `denied` event. Defaults: all read/search tools on;
  `run_tests`, `check_syntax`, `lint`, `git_status`, `git_diff`, `outline`,
  `find_symbol` on; `apply_patch` follows the write gates (refused in
  read-only). README section "Tools policy".

---

## Chunk B — verbs

### B1 `guru/domain/code.py` — `outline(path)` and `find_symbol(name, kind='')`
- `outline`: `ast` walk → one line per top-level and nested def/class:
  `L12-40 class Foo(Base)`, `  L14-22 def bar(self, x: int) -> str`; module
  docstring first line; digest capped (largest files: first 80 entries +
  "… N more"). Non-Python files: first 40 lines with line numbers.
- `find_symbol`: definitions via `ast` (functions, classes, assignments at
  module level) across allow-listed `.py` files under the project (skip
  noise dirs), references via a word-boundary grep; returns
  `def: path:line (kind)` lines then `ref: path:line: text` capped at 30;
  `kind` filters def/ref. Both gated by `ensure_path_allowed`.

### B2 `guru/domain/quality.py` — `run_tests`, `check_syntax`, `lint`
- `run_tests(target='', k='', maxfail=1, detail='')`: argv
  `[sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
  f'--maxfail={maxfail}', '-rfE', target?, '-k', k?]` (unittest runner:
  `[-m unittest, -q, target?]`); cwd = project root (first allow-listed
  read dir containing the target, else cwd). Digest: summary line, failing
  test ids (≤10) each with its assertion line; `detail=<test id>` returns
  that failure's full block (≤4 KB). Timeout → "timed out after Ns; N
  tests ran". Write-gated? No — tests read the repo; but refuse when the
  target path is outside the allow-list.
- `check_syntax(path)`: `py_compile.compile(path, doraise=True)` in-process
  (no subprocess) → "ok" or the SyntaxError line.
- `lint(path='')`: flake8 then mypy via fixed argv when configured in
  pyproject/setup.cfg/.flake8 (detect config presence; skip a linter that is
  not installed, say so); digest: counts + first 10 issues; `detail`
  expands.

### B3 `guru/domain/gitread.py` — `git_status()`, `git_diff(path='', detail=False)`
- Fixed argv (`git -C <root> status --porcelain=v1 -uall`,
  `git diff --stat` / `git diff -- <path>`); digest: changed-file list with
  +/- counts; `detail=True` returns the unified diff capped at 8 KB.
  Read-only: never `add`, `commit`, `checkout`.

### B4 `guru/domain/patch.py` — `apply_patch(diff)`
- Parses unified diffs for one or more files under the project; validates
  every hunk against the current file content (context must match; no fuzz)
  before writing anything; refuses new-file creation outside the project,
  renames and binary; goes through `files.ensure_write_path_allowed` per
  file and updates the sha ledger like `edit_file`; result digest: per-file
  hunks applied and new sha. All-or-nothing across files.

### B5 registry + prompts
- `TOOL_REGISTRY` entries with tags, parameters, `optional`, and `retain`
  policies (`run_tests`/`lint` → 'keep' (already small); `git_diff`
  detail → 'outline'-like cap). `PREACTIVATE_TOOLS` gains `outline`,
  `find_symbol`, `run_tests` so workers do not need a search hop.
- `CONTROLLER_HINT` and the worker/task text: "verify edits with
  `run_tests`/`check_syntax` before reporting; use `outline`/`find_symbol`
  before reading whole files".

---

## Chunk C — evaluation and docs

- Eval cases: `fix-failing-test` and `edit-then-verify` gain
  `tools_used_all = ["run_tests"]`; `guru-add-version-flag` gains
  `tools_used_all = ["run_tests"]`; new `find-symbol-outline` case (docs
  fixture-free: flaskish) expecting `outline`/`find_symbol` and `read_file`
  NOT used for the whole file; new `planted-failure-digest` case: cli-tool,
  prompt "Run the tests and tell me which test fails and why" →
  answer_regex ["newline", "test_words_across_newlines"], tools_used_all
  ["run_tests"], tools_used_none ["read_file"].
- README: "Audited tools" section (the verbs, digests, limits, policy file,
  `/tools`); docs/state-ownership.md bullets; design doc pointer.
- Real run: fast subset + edit cases + real guru cases on the Haiku
  controller config; compare pass rate, cost and `shown/produced` bytes with
  the 2026-09-24 runs; triage note.
