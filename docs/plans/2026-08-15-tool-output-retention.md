# Tool-Output Retention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the blanket per-turn tool-output prune with size-gated, per-tool retention (query-focused summary for web tools, code outline for large file reads, keep for everything else), configurable via `~/.guru/settings.toml`.

**Architecture:** Each `TOOL_REGISTRY` entry gains a `retain` policy (`keep` default, `summarize` for web tools, `outline` for `read_file`). `conversation.apply_retention()` replaces `prune_tool_exchanges` in `after_turn`: for each tool message from the finished turn it keeps small output verbatim and compacts only large output (LLM query-focused summary for `summarize`; AST/regex code skeleton for `outline`). The existing age-tiered `compact_messages` stays as the overflow backstop.

**Tech Stack:** Python 3.11+ (`ast`, `tomllib`), pytest, flake8 (79-col, PEP8/257/484). No new dependency.

**Project conventions (override the skill's defaults):**
- **Commit ONCE at the very end** (final task), not per task. Keep test-first cadence within each task.
- `.venv`: `source .venv/bin/activate` (or `.venv/bin/python`).
- `flake8 guru tests` clean and `pytest` green before the final commit.
- No emojis in code/output unless already present in that surface.
- Design doc `docs/plans/2026-08-15-tool-output-retention-design.md` already exists; fold it into the final commit.

---

### Task 1: Global settings file (`~/.guru/settings.toml`)

**Files:**
- Modify: `guru/config.py` (add `SETTINGS_PATH`, defaults, `load_settings`, `_apply_settings`, call at import)
- Test: `tests/test_guru.py` (new class `TestSettings`)

**Step 1: Write failing tests** (append to `tests/test_guru.py`):

```python
class TestSettings:
    """Global ~/.guru/settings.toml overrides context thresholds."""

    def test_load_settings_missing_returns_empty(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            config, 'SETTINGS_PATH', tmp_path / 'settings.toml')
        assert config.load_settings() == {}

    def test_load_settings_reads_context_section(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            "[context]\nweb_summarize_over_chars = 1234\n", encoding='utf-8')
        monkeypatch.setattr(config, 'SETTINGS_PATH', p)
        assert config.load_settings() == {'web_summarize_over_chars': 1234}

    def test_load_settings_invalid_returns_empty(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text("not = valid = toml", encoding='utf-8')
        monkeypatch.setattr(config, 'SETTINGS_PATH', p)
        assert config.load_settings() == {}

    def test_apply_settings_overrides_defaults(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            "[context]\nweb_summarize_over_chars = 999\n"
            "outline_file_over_chars = 111\n", encoding='utf-8')
        monkeypatch.setattr(config, 'SETTINGS_PATH', p)
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)
        monkeypatch.setattr(config, 'OUTLINE_FILE_OVER_CHARS', 8000)
        config._apply_settings()
        assert config.WEB_SUMMARIZE_OVER_CHARS == 999
        assert config.OUTLINE_FILE_OVER_CHARS == 111
```

**Step 2: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestSettings -q` → FAIL.

**Step 3: Implement** in `guru/config.py`.

Add near the other `GURU_HOME` paths (after `MODEL_CTX_PATH`):
```python
SETTINGS_PATH = GURU_HOME / 'settings.toml'          # global user settings
```

Add near the other context constants (after `KEEP_RECENT_GROUPS`):
```python
# Tool-output retention thresholds (chars). Overridable via settings.toml's
# [context] section. Below the threshold a tool result is kept verbatim; above
# it, web results are query-summarized and large code reads are outlined.
WEB_SUMMARIZE_OVER_CHARS = 6000
OUTLINE_FILE_OVER_CHARS = 8000
```

Add these functions (near `load_adapter_configs`, which shows the tomllib pattern):
```python
def load_settings() -> dict:
    """Return the [context] table from ~/.guru/settings.toml (or {})."""
    try:
        import tomllib
    except ModuleNotFoundError:                       # Python < 3.11
        import tomli as tomllib                        # type: ignore
    try:
        data = tomllib.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    section = data.get('context', {}) if isinstance(data, dict) else {}
    return section if isinstance(section, dict) else {}


def _apply_settings() -> None:
    """Override the retention thresholds from settings.toml, if present."""
    global WEB_SUMMARIZE_OVER_CHARS, OUTLINE_FILE_OVER_CHARS
    s = load_settings()
    try:
        WEB_SUMMARIZE_OVER_CHARS = int(
            s.get('web_summarize_over_chars', WEB_SUMMARIZE_OVER_CHARS))
        OUTLINE_FILE_OVER_CHARS = int(
            s.get('outline_file_over_chars', OUTLINE_FILE_OVER_CHARS))
    except (TypeError, ValueError):
        pass
```

At the bottom, after `ensure_setup()` and the `ALLOWED_*` loads, add:
```python
_apply_settings()
```

**Step 4: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestSettings -q` → PASS. `flake8 guru/config.py` clean.

---

### Task 2: `retain` policy on the tool registry

**Files:**
- Modify: `guru/domain/tools.py` (add `retain` to entries; add `retain_policy()`)
- Test: `tests/test_guru.py` (new class `TestRetainPolicy`)

**Step 1: Write failing tests:**

```python
class TestRetainPolicy:
    def test_web_tools_summarize(self) -> None:
        assert tools.retain_policy('web_search') == 'summarize'
        assert tools.retain_policy('web_fetch') == 'summarize'

    def test_read_file_outline(self) -> None:
        assert tools.retain_policy('read_file') == 'outline'

    def test_local_tools_keep(self) -> None:
        for name in ('search_code', 'list_dir', 'list_tree',
                     'write_file', 'edit_file', 'delete_file'):
            assert tools.retain_policy(name) == 'keep'

    def test_unknown_keeps(self) -> None:
        assert tools.retain_policy('nope') == 'keep'
        assert tools.retain_policy('') == 'keep'
```

**Step 2: Run** → FAIL (`retain_policy` missing).

**Step 3: Implement** in `guru/domain/tools.py`.

Add `"retain": "summarize",` to the `web_search` and `web_fetch` registry entries, and `"retain": "outline",` to the `read_file` entry. (Leave other entries alone — they default to keep.)

Add this function (near `active_specs`/`specs_for`):
```python
def retain_policy(name: str) -> str:
    """How a tool's output should be retained after a turn: 'keep' (default),
    'summarize' (query-focused, for bulky web output), or 'outline' (code
    skeleton, for large file reads)."""
    info = TOOL_REGISTRY.get(name)
    if not info:
        return 'keep'
    return info.get('retain', 'keep')
```

Note: `retain` is metadata only — `specs_for` builds specs from `name`/`description`/`parameters`/`optional`, so `retain` is never sent to the model. No change needed there.

**Step 4: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestRetainPolicy -q` → PASS. `flake8 guru/domain/tools.py` clean.

---

### Task 3: Query-focused summary helper

**Files:**
- Modify: `guru/domain/conversation.py` (add `_recent_question`, `_summarize_relevant`)
- Test: `tests/test_guru.py` (new class `TestFocusedSummary`)

**Step 1: Write failing tests:**

```python
class TestFocusedSummary:
    def test_recent_question_scans_backwards(self) -> None:
        msgs = [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': 'a'},
            {'role': 'user', 'content': 'the question'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'x'},
        ]
        assert conversation._recent_question(msgs, 4) == 'the question'

    def test_summarize_relevant_uses_adapter(self, monkeypatch) -> None:
        seen = {}

        class _Ad:
            def summarise(self, transcript):
                seen['t'] = transcript
                return 'RELEVANT BITS'
        monkeypatch.setattr(session, 'adapter', _Ad())
        out = conversation._summarize_relevant(
            'k8s setting?', 'huge content', 'web_fetch')
        assert 'RELEVANT BITS' in out
        assert 'web_fetch summary' in out and 'k8s setting?' in out
        assert 'huge content' in seen['t'] and 'k8s setting?' in seen['t']

    def test_summarize_relevant_falls_back_on_error(
            self, monkeypatch) -> None:
        class _Ad:
            def summarise(self, transcript):
                raise RuntimeError('down')
        monkeypatch.setattr(session, 'adapter', _Ad())
        out = conversation._summarize_relevant('q', 'A' * 100, 'web_fetch')
        assert 'truncated' in out and out.count('A') > 0
```

**Step 2: Run** → FAIL.

**Step 3: Implement** in `guru/domain/conversation.py` (module already imports `config`, `session`):

```python
def _recent_question(messages: list, index: int) -> str:
    """The most recent user message at or before ``index`` (the question the
    tool output was helping answer)."""
    for j in range(min(index, len(messages) - 1), -1, -1):
        if msg_role(messages[j]) == 'user':
            return msg_content(messages[j]).strip()
    return ''


def _summarize_relevant(question: str, content: str, tool_name: str) -> str:
    """Query-focused compaction: ask the model to keep only the parts of a
    bulky tool result relevant to the question. Falls back to truncation."""
    prompt = (
        "Extract only the parts of the following content that are relevant to"
        f" answering this question. Keep concrete facts, values, and"
        f" identifiers; be concise; output only the extract.\n\n"
        f"Question: {question}\n\nContent:\n{content}")
    try:
        summary = (session.adapter.summarise(prompt) or '').strip()
    except Exception as e:                            # noqa: BLE001
        summary = ''
        _ = e
    if not summary:
        summary = content[:config.WEB_SUMMARIZE_OVER_CHARS] + "\n…(truncated)"
    q = question[:60]
    return f"[{tool_name} summary · query: {q}]\n{summary}"
```

**Step 4: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestFocusedSummary -q` → PASS. `flake8` clean.

---

### Task 4: Code outline helper

**Files:**
- Modify: `guru/domain/conversation.py` (add `import ast`, `import re`; add `_outline_code` + `_sig`)
- Test: `tests/test_guru.py` (new class `TestOutlineCode`)

Context: a `read_file` result is a header line `"<path> (lines A-B of N, sha:xxx):"` followed by numbered body lines `"{lineno:>6}\t{source}"`.

**Step 1: Write failing tests:**

```python
class TestOutlineCode:
    def _read_output(self, path, source):
        lines = source.splitlines()
        header = f"{path} (lines 1-{len(lines)} of {len(lines)}, sha:abc):"
        body = "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(lines, 1))
        return f"{header}\n{body}"

    def test_outline_python_keeps_signatures_drops_bodies(self) -> None:
        src = (
            "import os\n"
            "\n"
            "def foo(a, b=2) -> int:\n"
            "    '''Add things.'''\n"
            "    return a + b\n"
            "\n"
            "class C:\n"
            "    def method(self, x):\n"
            "        '''Do it.'''\n"
            "        return x\n")
        out = conversation._outline_code(
            self._read_output('/tmp/m.py', src))
        assert 'def foo(a, b=2) -> int:' in out
        assert 'Add things.' in out
        assert 'class C' in out
        assert 'def method(self, x):' in out
        assert 'return a + b' not in out      # body dropped
        assert '/tmp/m.py' in out             # header kept

    def test_outline_non_python_truncates(self) -> None:
        src = "\n".join(f"line {i}" for i in range(200))
        out = conversation._outline_code(
            self._read_output('/tmp/notes.txt', src))
        assert '/tmp/notes.txt' in out
        assert len(out) < len(src)            # shrunk

    def test_outline_unparseable_python_falls_back(self) -> None:
        src = "def broken(:\n    pass\nimport sys\n"
        out = conversation._outline_code(
            self._read_output('/tmp/b.py', src))
        # regex fallback keeps def/import lines even when AST fails
        assert 'import sys' in out or 'def broken' in out
```

**Step 2: Run** → FAIL.

**Step 3: Implement** in `guru/domain/conversation.py`. Add `import ast` and `import re` to the imports at the top. Then:

```python
_OUTLINE_RE = re.compile(r'^\s*(async def |def |class |@|import |from )')
_NUM_PREFIX = re.compile(r'^\s*\d+\t')


def _sig(node) -> str:
    """A def/class signature line without the body."""
    if isinstance(node, ast.ClassDef):
        bases = ', '.join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    prefix = 'async def ' if isinstance(node, ast.AsyncFunctionDef) else 'def '
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ''
    return f"{prefix}{node.name}({ast.unparse(node.args)}){ret}:"


def _outline_code(read_output: str) -> str:
    """Compact a large read_file result to a navigable skeleton: the header,
    imports, and def/class signatures + one-line docstrings (bodies dropped).
    AST for a full .py read; regex fallback for partial reads / non-Python."""
    parts = read_output.split('\n')
    header = parts[0] if parts else ''
    body = parts[1:]
    path = header.split(' (lines', 1)[0].strip()
    source = "\n".join(_NUM_PREFIX.sub('', ln) for ln in body)

    out = [header, '[outline]']
    if path.endswith('.py'):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            doc = ast.get_docstring(tree)
            if doc:
                out.append(f'"""{doc.splitlines()[0]}"""')
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    out.append(ast.unparse(node))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef)):
                    out.append(_sig(node))
                    d = ast.get_docstring(node)
                    if d:
                        out.append(f'    """{d.splitlines()[0]}"""')
                    if isinstance(node, ast.ClassDef):
                        for sub in node.body:
                            if isinstance(sub, (ast.FunctionDef,
                                                ast.AsyncFunctionDef)):
                                out.append('    ' + _sig(sub))
                                sd = ast.get_docstring(sub)
                                if sd:
                                    out.append(
                                        f'        """{sd.splitlines()[0]}"""')
            return "\n".join(out)

    # Fallback: keep structural lines (numbered, for navigation), else head.
    kept = [ln for ln in body if _OUTLINE_RE.match(_NUM_PREFIX.sub('', ln))]
    if len(kept) >= 3:
        return "\n".join([header, '[outline]'] + kept)
    return "\n".join([header, '[truncated]'] + body[:40])
```

**Step 4: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestOutlineCode -q` → PASS. `flake8` clean (watch 79-col in `_sig`).

---

### Task 5: `apply_retention` + wire into `after_turn`

**Files:**
- Modify: `guru/domain/conversation.py` (add `apply_retention`; call it in `after_turn`; remove `prune_tool_exchanges`)
- Modify: `tests/test_guru.py` (remove `TestPruneToolExchanges`; add `TestApplyRetention`)

**Step 1: Check for other callers**

Run: `grep -rn prune_tool_exchanges guru tests`
It should appear only in `conversation.py` (definition + `after_turn` call) and `tests/test_guru.py::TestPruneToolExchanges`. If it appears elsewhere, keep the function; otherwise remove it in Step 3.

**Step 2: Write failing tests** (add `TestApplyRetention`; delete the old `TestPruneToolExchanges` class):

```python
class TestApplyRetention:
    def _msgs(self):
        return [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'find the k8s setting'},
            {'role': 'assistant', 'content': ''},        # text-less step
            {'role': 'tool', 'tool_name': 'web_fetch',
             'content': 'X' * 9000},                      # large -> summarize
            {'role': 'tool', 'tool_name': 'search_code',
             'content': 'a.py:1: hit'},                   # keep
            {'role': 'assistant', 'content': 'the answer'},
        ]

    def test_large_web_summarized_small_kept(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)

        class _Ad:
            def summarise(self, t):
                return 'GIST'
        monkeypatch.setattr(session, 'adapter', _Ad())
        msgs = self._msgs()
        conversation.apply_retention(msgs)
        # text-less assistant dropped
        assert all(not (m.get('role') == 'assistant'
                        and not m.get('content')) for m in msgs)
        web = next(m for m in msgs if m.get('tool_name') == 'web_fetch')
        assert 'GIST' in web['content'] and len(web['content']) < 9000
        grep = next(m for m in msgs if m.get('tool_name') == 'search_code')
        assert grep['content'] == 'a.py:1: hit'          # untouched

    def test_small_web_not_summarized(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)

        class _Ad:
            def summarise(self, t):
                raise AssertionError('should not summarise small output')
        monkeypatch.setattr(session, 'adapter', _Ad())
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'short'},
        ]
        conversation.apply_retention(msgs)
        assert msgs[-1]['content'] == 'short'

    def test_large_read_file_outlined(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'OUTLINE_FILE_OVER_CHARS', 50)
        src = "def foo():\n    return 1\n" * 10
        header = f"/tmp/m.py (lines 1-20 of 20, sha:abc):"
        body = "\n".join(f"{i:>6}\t{ln}"
                         for i, ln in enumerate(src.splitlines(), 1))
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_name': 'read_file',
             'content': f"{header}\n{body}"},
        ]
        conversation.apply_retention(msgs)
        assert '[outline]' in msgs[-1]['content']
        assert 'return 1' not in msgs[-1]['content']
```

**Step 3: Implement** in `guru/domain/conversation.py`:

```python
def apply_retention(messages: list) -> None:
    """Post-turn retention: drop text-less tool-call steps, then compact each
    tool result per its tool's retain policy (keep / summarize / outline),
    only when it exceeds the size threshold. Replaces the blanket prune so
    follow-up questions keep the useful, relevant context."""
    kept = []
    for i, m in enumerate(messages):
        role = msg_role(m)
        if role == 'assistant' and not msg_content(m).strip():
            continue
        if role == 'tool' and isinstance(m, dict):
            name = m.get('tool_name', '')
            policy = tools.retain_policy(name)
            content = m.get('content') or ''
            if (policy == 'summarize'
                    and len(content) > config.WEB_SUMMARIZE_OVER_CHARS):
                q = _recent_question(messages, i)
                m['content'] = _summarize_relevant(q, content, name)
            elif (policy == 'outline'
                    and len(content) > config.OUTLINE_FILE_OVER_CHARS):
                m['content'] = _outline_code(content)
        kept.append(m)
    messages[:] = kept
```

In `after_turn`, replace `prune_tool_exchanges(session.messages)` with:
```python
    apply_retention(session.messages)
```

Then remove the now-unused `prune_tool_exchanges` function (confirmed in Step 1). Update its docstring reference in `after_turn` ("Prune tool output" → "Apply per-tool retention").

**Step 4: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestApplyRetention -q` → PASS. Then the full suite `.venv/bin/python -m pytest -q` → green (minus the removed `TestPruneToolExchanges`). `flake8 guru tests` clean. `.venv/bin/python -c "import guru.domain.conversation, guru.tui"` → OK.

---

### Task 6: Manual smoke + final verification + single commit

**Step 1: Manual smoke** (`./start.sh`):
- Ask something requiring a `web_fetch` of a large page, then a follow-up — confirm the follow-up still has the relevant gist (a `[web_fetch summary · query: …]` block) and answers without re-fetching.
- `read_file` a large `.py`, follow up — confirm an `[outline]` (signatures + docstrings) is retained.
- `read_file` a small file — confirm it's kept verbatim.
- Optionally drop a `~/.guru/settings.toml` with `[context]\nweb_summarize_over_chars = 2000` and confirm the threshold changes.

**Step 2:** `source .venv/bin/activate && flake8 guru tests` → clean; `pytest -q` → all green.

**Step 3: Single commit** (includes the design doc):
```bash
git add -A
git commit -m "feat: per-tool output retention (query-summary for web, outline for large reads)"
```
(No AI/LLM mention in the message/body.)

**Step 4:** Remind the user to `ssh-add ~/.ssh/id_ed25519` then `git push origin main`.

---

## Notes for the executor
- **DRY/YAGNI:** the query-focused summary reuses `adapter.summarise` (no new adapter method); no narrower code-search tools in this change (explicitly deferred).
- **Fallbacks never lose a turn:** summary failure → truncate; outline parse failure → regex/head.
- **Thresholds are read from `config` at call time**, so tests monkeypatch `config.WEB_SUMMARIZE_OVER_CHARS` / `config.OUTLINE_FILE_OVER_CHARS`.
- Age-tiered `compact_messages` is intentionally untouched — it's the overflow backstop and now has retained tool output to work on.
