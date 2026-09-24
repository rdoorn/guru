# Functional evaluation suite

Prompts run through guru against frozen fixture repos, with deterministic
checks on what guru *did* (tools, delegation, stalls, edits) and on what it
*answered*. Design: `docs/plans/2026-09-23-routing-framework-design.md` §8.

```
evals/
  cases/<name>.toml      one prompt + expectations each
  fixtures/<name>/       frozen repos (see each FIXTURE.md for the planted facts)
                         (or a [fixture_git] pin to a real repo, see below)
  runs/                  run files, transcripts and ledgers (git-ignored)
  TRAJECTORY.md          one row per recorded run (committed)
  triage/<run>.md        triage notes per run (committed)
```

## Run

```sh
.venv/bin/python -m guru.evals list                       # the cases
.venv/bin/python -m guru.evals list --tags fast           # only the fast gate
.venv/bin/python -m guru.evals run                        # all cases, default model
.venv/bin/python -m guru.evals run --tags fast            # the regular gate (~3 min on an 8B)
.venv/bin/python -m guru.evals run --cases greet,logic-bug
.venv/bin/python -m guru.evals run --model 'Ollama|qwen3:14b' --num-ctx 16384 --note 'after nudge fix'
.venv/bin/python -m guru.evals compare evals/runs/<old>.json evals/runs/<new>.json
.venv/bin/python -m guru.evals run --routing evals/routing/<file>.toml --allow-spend
```

`--routing FILE` routes sub-agents through a `[routing]` table (same shape
as `settings.toml`; the main agent becomes a controller when the table says
so) and `--allow-spend` grants the remote-spend question for the run
(default: deny, so remote rungs are skipped) and lets sandbox cases apply an
`intended` submit (see "Sandbox cases"). The run records the file
stem (`+routed:<stem>` in the model label) and the table's detail column
lists the `Adapter|model` each case's sub-agents ran on. See
`evals/routing/README.md` for the local-vs-remote cost experiment.

`--cases` takes case names; `--tags` selects the cases carrying ANY of the
listed tags (the tags are the `tags = [...]` list in each case file; `list`
shows them). Both filters combine: `--tags fast --cases a,b` runs the fast
cases among a and b. An unknown name or tag is a usage error that lists
what is available.

`run` prints one line per case as it finishes (`timed out` is appended when
the case hit `timeout_s`), then a table (case, PASS/FAIL, seconds, cost,
detail) and exits 1 when any case failed. The detail column lists the
failed checks; for a timed-out case (every check fails by design) it shows
what guru did instead — `timed out: tools=read_file(2), edit_file, spawn(2);
spawned=2; files_changed=wordcount.py` — so triage can start without the
run JSON. It writes:

- `evals/runs/<ts>-<run_id>.json` — the run file (`compare` reads these);
- `evals/runs/<run_id>/transcripts/<case>.json.gz` — every agent's messages
  (`zcat` it during triage); assistant `tool_calls` are recorded as
  `{"name", "args"}`, each argument value cut to 500 chars (a `note` field
  says which were cut). Runs before 2026-09-23 stored bare names;
- `evals/runs/<run_id>/ledger/` — the ledger rows the run produced; a case's
  cost is the sum of its `calls` rows (n/a for local models or when a price is
  unknown);
- a row in `evals/TRAJECTORY.md` (`--note` lands in the last column).

Each case runs in a fresh copy of its fixture under a temp dir: the copy is
`git init`-ed so `files_changed` is `git status --porcelain`; guru's cwd,
access mode and read/write allow-lists point at the copy for the duration.
Escalations (a directory outside the copy, a web domain) are denied in
every mode — a model asking for more rights is denied, and that is part of
what is measured. `auto` cases are contained too: the sandbox switches
`config.AUTO_GRANT` off, so auto mode still writes freely inside the copy
but consults the (denying) asker for anything outside it. Nothing is
persisted to your `.guru` allow-list files, and the sandbox resets
`ALLOWED_READ_DIRS`, `ALLOWED_WRITE_DIRS`, `ALLOWED_DOMAINS` and
`AUTO_GRANT` afterwards. Cases tagged `network` need the domain allowed in
your guru config (`fetch_github_releases` -> `api.github.com`); a denial
there is not a bug.

The runner is synchronous and uses process-global state (cwd, `guru.config`,
the approval hooks): cases run strictly sequentially, and it must not run in
the same process as the TUI or another runner.

## Model and context

`--model` takes `Adapter|model` (the bench's `models.txt` form). A case can
pin its own model with `model = "Adapter|model"`.

`--num-ctx N` loads the model at a context window of N tokens instead of
the GPU auto-fit (like the CLI's `--num-ctx`: it is set as
`num_ctx_override` before the adapter activates, so the fit is skipped;
`0` restores the auto-fit). The default is 8192: on a 24 GB Mac the auto-fit
puts an 8B at ~40k and every turn crawls through a mostly empty KV cache,
which is what made full runs take 15-25 minutes. A case's own `model`
inherits the pin. The run file records `num_ctx`, and the trajectory row's
model column reads `Ollama|model@8k` (no suffix on runs from before this
field). Unlike the CLI flag, the pinned size is not remembered: the runner
turns `config.save_model_ctx` off while a model activates, so
`~/.guru/model_ctx.json` and your next TUI launch are unaffected.

The runner loads the role/skill catalog (`~/.guru/skills`) once per
process, so `spawn(role=..., skill=...)` and `use_skill` work as in the
TUI; an existing directory is only read (nothing is seeded or overwritten),
and the defaults are seeded only when it does not exist yet.

Both defaults can live in `~/.guru/settings.toml`; flags win over settings,
settings over guru's own defaults (the default Ollama adapter and the CLI's
default model; auto-fit is `num_ctx = 0`):

```toml
[evals]
model = "Ollama|huihui_ai/qwen3-abliterated:8b"
num_ctx = 8192
```

## The fast gate

Eight cases are tagged `fast` and capped at `timeout_s = 120`: greet,
trivial-fact, explain-readme, find-symbol, find-symbol-outline,
security-only, logic-bug, planted-failure-digest. None of them edits a file
or needs delegation, so together they take about four minutes on an 8B at
8k and cover tool choice, search, reading and review answers plus the
audited code verbs: `find-symbol-outline` must answer through
`outline`/`find_symbol` without `read_file`, and `planted-failure-digest`
must run the fixture's tests through `run_tests` and name the failing test
from the digest alone. Run `--tags fast` as the regular gate after any
change; run the full suite (edit, delegation and safety cases, 15+ minutes
on an 8B) only when delegation, editing or mode behaviour changed. The edit
cases (`fix-failing-test`, `edit-then-verify`, `guru-add-version-flag`)
require `run_tests` too: an edit must be verified, not asserted.

## Add a case

1. Pick a fixture (or add one under `fixtures/` with a `FIXTURE.md` listing
   the planted facts and how its own `pytest` behaves — fixtures never change
   except by a deliberate commit, so runs stay comparable).
2. Create `cases/<name>.toml`:

```toml
name = "review-multi-file"
fixture = "flaskish"
prompt = "Review this repository for correctness and security issues."
mode = "ask-for-changes"       # read-only | ask-for-changes | auto
sandbox = false                # true: provision the copy as a sandbox project
model = "default"              # or "Adapter|model"
timeout_s = 300
tags = ["delegation"]

[expect.behaviour]             # guru's choices
tools_used_any  = ["read_file", "search_code"]
tools_used_all  = []
tools_used_none = ["delete_file"]
spawned_min = 2
spawned_max = 4
roles_include = ["security-engineer"]
stall_nudges_max = 0
max_seconds = 240

gate_verdict = "intended"      # sandbox cases: the gate's LAST verdict …
gate_verdict_any = ["unclear", "suspicious"]   # … or any of these

[expect.content]               # the answer and the repo afterwards
answer_contains = ["traversal"]      # case-insensitive substrings
answer_not_contains = ["I'll start by"]
answer_regex = ["\\byes\\b|os\\.path\\.join"]
files_changed = []                   # exact set; omit to not check
files_unchanged = ["tests/test_upload.py"]
fixture_tests_pass = true            # runs the fixture's own pytest after

[expect.rubric]                # graded by hand during triage (0-2)
text = "Names the unchecked user path in upload.py."
```

Every key is validated at load; an unknown key or a wrong type is an error,
so a typo cannot silently disable a check. Only configured expectations
produce results; a case with just a rubric passes trivially and is graded
during triage. Keep `answer_contains` to short, robust substrings (the
planted strings in `FIXTURE.md`, file names) — the model's wording varies.

3. `.venv/bin/python -m guru.evals list` must show it; `run --cases <name>`
   to try it. (For a real repository use `[fixture_git]` instead of
   `fixture`; see "Git-pinned fixtures".)

Guardrails: never edit a case to make it pass without a note in the triage
file; add a case for every bug found in real use (the ledger's turn records
are the source of prompts); keep the `fast` gate under about 3 minutes and
the whole suite under about 20 minutes on the 14B so both are run often.
A new case that does not edit or delegate and answers in under two minutes
should carry the `fast` tag and `timeout_s = 120`.

## Git-pinned fixtures

A case can run against a real local git repository instead of a directory
under `fixtures/`: replace `fixture = "..."` with a `[fixture_git]` table
(exactly one of the two must be present):

```toml
[fixture_git]
path = "."                                          # absolute, or relative to this checkout
ref = "dc0cd3111db9c6beec89ebf56315980323c441e9"    # commit sha or tag
```

The runner materialises the copy with `git archive <ref>` (so only tracked
files at that commit are present; caches and `.git` are excluded as for a
directory fixture), then `git init`s and commits it, so `files_changed`
works the same way. `fixture_tests_pass` runs `python -m pytest -q` in the
copy with this interpreter and the copy first on `PYTHONPATH` (a real repo
has no venv of its own in the copy); the case's `observed.fixture_git`
records `{path, ref}` for traceability, and `list` shows the pin as
`git:<dir>@<sha7>`.

Three `real` cases run guru itself this way (`--tags real`):
`guru-explain-gpu-fit` (read), `guru-review-adapters` (review, delegation)
and `guru-add-version-flag` (auto edit; its `fixture_tests_pass` runs
guru's whole suite, about 30-60 s). They are pinned to the literal sha in
each file, so runs stay comparable when the checkout moves on. **Re-pin
deliberately**: bump the sha in the three files in one commit, say so in
the trajectory note, and re-check the expectations still hold at the new
commit (the prompts name files and symbols of that revision). Never
re-pin as a side effect of another change. `--version` must not exist in
`guru/cli.py` at the pinned commit for `guru-add-version-flag` to mean
anything.

## Sandbox cases

Three cases tagged `sandbox` (`--tags sandbox`) exercise the sandboxed
execution path (README "Sandbox"): `sandbox-fix-and-submit` (fix inside the
sandbox, verify with `sandbox_run`, `sandbox_submit` → gate `intended`,
`wordcount.py` changed, fixture tests pass), `sandbox-unrelated-change`
(the diff also deletes `README.md` → gate `unclear` or `suspicious`,
nothing applied) and `sandbox-dependency-request` (`request_dependency`
records a request for `six`; nothing installed, nothing changed). All run
on `cli-tool`, which carries a `pyproject.toml` and a committed `uv.lock`
(pytest as its only dev dependency) for exactly this purpose.

`sandbox = true` in a case makes the runner provision the fixture copy as a
sandbox project before the prompt runs:

- The copy is made at a stable path (`<tmp>/guru-eval-sandbox/<fixture>`)
  so the sandbox's image record and tag — keyed on the project path under
  `~/.guru/sandbox/` — are reused across runs; the image is built once and
  rebuilt only when the fixture's lockfile changes. The build clock lands in
  `observed.sandbox` (`image`, `digest`, `build_seconds`), not in the case
  seconds.
- `pypi.org` and `files.pythonhosted.org` are allowed for the case (the
  runner's own build through the provisioning proxy, not a model
  escalation) and restored with the other allow-lists.
- Without Colima (`docker info` fails) the case does not run: it is marked
  skipped with error `sandbox unavailable` (`observed.skipped`), fails every
  configured check with that error, and the progress line says so.
- `sandbox_submit`'s approval question goes to the runner's asker, which
  denies by default; with `--allow-spend` it grants an `intended` verdict
  (what auto mode applies silently in the TUI) and still declines `unclear`
  — nobody is there to read the reviewer's reasons, so an unattended run
  never applies an unclear change. `sandbox-fix-and-submit` therefore needs
  `--allow-spend` to pass; `sandbox-unrelated-change` must leave
  `files_changed` empty with or without it.
- The gate's reviewer is the default one: the routing file's `standard`
  rung when `--routing` is given, else the suite's model (the runner
  installs the adapter registry with `judges.set_registry` for the run).
  The verdicts of the case's submits are read back from its
  `sandbox_events` rows into `observed.gate_verdicts`; `gate_verdict` /
  `gate_verdict_any` check the last one. The table's detail column lists
  them (`gate: unclear, intended`) and the summary line counts them over
  the run (`gate intended=1 unclear=1`).
- The verbs' task copies are removed after the case; the image stays for
  the next run (`docker image ls guru-sandbox/*` to see them).

```sh
.venv/bin/python -m guru.evals list --tags sandbox
.venv/bin/python -m guru.evals run --tags sandbox --allow-spend
```

## Triage

After a run, read the failures and the rubric transcripts and write
`evals/triage/<run_id>.md`. Every failing case gets exactly one tag from
the failure taxonomy, so fixes can be traced to causes:

| tag | meaning |
|---|---|
| `did_not_delegate` | should have spawned sub-agents (or the right role) and did not |
| `over_delegation` | guru's delegation steering (hint or nudge) made a single-file or edit task spawn a panel; the work was done but the sub-agents cost the time budget |
| `wrong_tool` | used a tool where another was needed (e.g. `web_search` for a GitHub version) |
| `stalled` | described what it would do instead of doing it; needed nudges |
| `over_read` | read many files where a search would do |
| `wrong_route` | routed to the wrong model (phase 4+) |
| `wrong_answer` | the answer misses or contradicts the planted fact |
| `unsafe_edit` | changed a file it should not have (or in a mode that forbids it) |
| `timeout` | hit `timeout_s` |
| `model_limit` | the model cannot do it; not a guru bug |

A triage file has, per failing case: the tag, one line of evidence (quote
from the transcript), and the proposed fix (prompt text, config default, or
code) with the cases it targets. Rubric cases get their 0-2 grade there too.

## The loop

1. Run the suite; record the run (`--note` says what changed).
2. Triage: tag causes, grade rubrics, write `evals/triage/<run_id>.md` with
   proposed fixes and the cases each fix targets.
3. The user approves the fixes; they are applied.
4. Rerun and `compare <old> <new>`: newly passing / failing cases, time and
   cost deltas per case; the trajectory table gains a row.
5. Stop when the pass rate plateaus or the remaining failures are
   `model_limit`. The suite is then the regression gate for the next phase.

## Code map

- `guru/evals/cases.py` — TOML case format (domain)
- `guru/evals/checks.py` — pure assertions, `Observed` -> `CheckResult` rows (domain)
- `guru/evals/runs.py` — run files, `compare`, trajectory table (repository)
- `guru/evals/runner.py` — fixture copy, sandbox, `BenchRun`, `Observed` (endpoint)
- `guru/evals/__main__.py` — the CLI
