# Guru roles & skills — design

Date: 2026-08-15

## Purpose

Give guru pre-defined specialized behaviors that stay useful on small
local models with tight context windows. Two orthogonal, composable axes —
a persona and a procedure — that the model can self-select or the user can
force, without a 1:1 copy of larger agents' verbose prompts.

## The distinction (baseline)

Two orthogonal axes:

- **Role = identity / persona (WHO).** A standing point of view: domain
  expertise, priorities, vocabulary, what it notices. Nouns. Sticky across
  many turns. Sets the *lens*. Default is `general-purpose` (implicit; no
  overlay). Examples: developer, architect, security-engineer, SRE.
- **Skill = procedure / methodology (HOW).** A repeatable technique,
  independent of who you are. Gerunds/verbs. Sticky-single, swaps as the
  task phase changes. Sets the *procedure*. Examples: code-review,
  brainstorming, systematic-debugging, test-driven-development.

Classifier test: *"Can two different personas both do X?"* If yes, X is a
**skill** (a developer, architect, and security engineer can all do a code
review, emphasizing different things). So `code-review` is a skill, not a
role.

They **multiply**: `role × skill` = a persona applying a technique. Same
skill, different lens (security-engineer + code-review emphasizes injection
/ authz / secrets; architect + code-review emphasizes boundaries /
coupling / scalability). N roles × M skills behaviors from N+M small files.

Explicitly **out of scope for v1**: read-only "explore / fan-out". That is a
third axis (execution mode / permissions), not a persona or procedure.
Deferred. Consequently v1 roles and skills are **pure prompt overlays** —
they do not restrict tools; tool access stays governed by the existing
read-only / ask / auto access mode.

## Model

- Effective system prompt per agent =
  `base (built-in + GURU.md)` + `[role block]?` + `[skill block]?`.
- **At most one role and one skill active at a time** (≤ 2 overlays), so the
  added context is bounded.

## Storage & seeding

One directory, both kinds, disambiguated by `kind`:

```
~/.guru/skills/<name>.md
---
name: security-engineer
kind: role                 # role | skill
description: <one line — used for the catalog and model selection>
model:                     # optional model override; omit = inherit
---
<body: compact persona or procedure, <= 512 tokens>
```

- Baked-in defaults are defined in a code dict. On startup, any default whose
  file is **absent** is written out. User edits are kept.
- `--reset-skills` overwrites the baked-in defaults (roles + skills) on
  startup so refinements ship; user-authored extra files are untouched.
- Body budget: **<= 512 tokens**, enforced as a hard cap at load (a longer
  body is truncated with a `…[truncated]` marker) so an over-long user edit
  cannot blow the window. Defaults authored to sit under it.

## Runtime state & residency

- `SessionState` gains `active_role: str | None`, `active_skill: str | None`
  (per-agent, so sub-agents carry their own).
- A single `refresh_system_context()` rebuilds the dynamic tail of
  `messages[0]` each turn, **absorbing the current `[open files]` ledger**
  and adding:
  - the **catalog** — grouped Roles / Skills, one `name — description` line
    each (lets the model self-select);
  - a `[role: X]` block (the role body) when a role is active;
  - a `[skill: Y]` block (the skill body) when a skill is active.
  Each is a single copy, survives pruning/compaction (lives on message 0),
  and is counted in the `sys:` context bucket. Replaces `refresh_file_ledger`
  and is called in the same spot in `_work`.
- Catalog cost ~ 8 lines (~160 tokens) resident; overlays add <= 2 bodies
  (worst case ~1k tokens, only when both are active).

## Tools & commands

- Model-facing:
  - `use_skill(name)` — set the active skill on the current agent.
  - `spawn(role=?, skill=?, task=…)` — both optional; the child runs isolated
    with that persona + procedure and returns only its conclusion (existing
    mailbox flow).
- User-facing:
  - `/role <name>` / `/role off` (off = general-purpose) — re-skins the
    current agent.
  - `/skill <name>` / `/skill off` — sets/clears the current agent's skill.
- The catalog lets the model self-select; the commands are the reliable
  override for when a small model does not pick well. The model never
  re-skins itself — it delegates via `spawn`. Added model surface is just
  `use_skill` plus two `spawn` params.

## Wiring

- Registry loaded once at startup (new `guru/skills.py`, or in `config`):
  parse `~/.guru/skills/*.md` into `{name: entry}`.
- `_configure(agent, base, can_spawn, role=None, skill=None)` sets
  `active_role` / `active_skill`.
- `refresh_system_context()` resolves active names -> bodies from the registry
  and renders the overlays.
- `_spawn_agent(task, role=None, skill=None)` threads both through
  `_configure`.

## Defaults shipped

- **Roles:** `developer`, `architect`, `security-engineer`, `SRE`
  (+ implicit `general-purpose`).
- **Skills:** `code-review`, `brainstorming`, `systematic-debugging`,
  `test-driven-development`.
- Each body a tight <= 512-token distillation of the persona / methodology —
  not a 1:1 copy of a larger agent's prompt.

## Error handling

- Unknown name -> the tool / command returns the available names.
- Malformed frontmatter -> skip that file with a startup warning.
- Wrong kind (a role where a skill is expected, or vice-versa) -> rejected
  with the correct list.

## Testing

- Registry parse (frontmatter, kind, model override).
- Seeding writes missing default files; `--reset-skills` overwrites defaults
  but not user-authored extras.
- 512-token body cap truncates.
- Catalog + role + skill render onto `messages[0]`, idempotent across turns,
  survive pruning, single copy each.
- Role / skill set / swap / clear via `/role`, `/skill`, `use_skill`.
- `spawn(role, skill)` yields a child whose `messages[0]` carries both
  overlays.
- Unknown-name and wrong-kind handling.

## Open items deferred

- Read-only "explore / fan-out" execution mode (and per-role/skill tool
  restriction) — a separate third axis, revisited after the core lands.
- `Ctrl+N` new-agent picking a role at creation time.
