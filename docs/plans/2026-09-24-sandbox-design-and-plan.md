# Sandboxed execution (Colima) with a lockfile-gated provisioning phase and an AI quality gate — design and plan

Status: S1–S4 implemented on `feat/audited-tools` (2026-09-24): runtime,
provisioning proxy, verbs + gate, eval cases and docs. Outstanding: a real
end-to-end run of the three `sandbox` eval cases against Colima
(`python -m guru.evals run --tags sandbox --allow-spend`) and its triage.
Builds on `2026-09-24-audited-tools-plan.md` (fixed-argv runner,
tool_events, tools policy, apply_patch).

## 0. Decisions

| # | Decision |
|---|---|
| 1 | Runtime: Docker CLI against Colima (Apple silicon). Apple's `container` runtime is a later option behind the same interface. |
| 2 | Execution phase runs with `--network none`. Provisioning runs on a Docker `--internal` network whose only other member is a filtering proxy container. An internal network has no external route and no external DNS forwarding, so the internet is unreachable except through the proxy; proxy env vars are a convenience, not the control. What *does* remain reachable from an internal network: the Colima VM's own listeners on the network's gateway address (e.g. the VM's sshd) — published container ports are not. The build container (classic builder) is less confined than the run container: it runs as root with default capabilities and a writable root, so the proxy allow-list and the lockfile are the controls during provisioning. |
| 3 | The proxy's host allow-list is generated from guru's existing project domain allow-list (`.guru/domains_allow.txt`). No second list. Adding `pypi.org` / `files.pythonhosted.org` is the normal "allow web access?" approval. CONNECT to allow-listed hosts on 443 only; plain HTTP refused; every request logged as `net_events`. The proxy is a third-party image (`docker.io/kalaksi/tinyproxy`, digest-pinned, runs unprivileged, read-only, caps dropped, on the internal network plus the default bridge). Accepted risk: a malicious image could relay anything for a client that reaches it — the client must still be inside the internal network and the image is pinned by digest, so the exposure is an upstream compromise of that exact digest. Follow-up: build guru's own proxy image from the official `alpine` digest with `apk add tinyproxy` so the only third party is Alpine. |
| 4 | Packages: only lockfile-declared dependencies exist in the image. The model never installs. `request_dependency(name, constraint)` records a request; on approval guru runs `uv add`/`uv lock` in the provision container and rebuilds. Lockfile diff is part of the approval. |
| 5 | The sandbox works on a copy of the project; nothing it does touches the real tree. Changes come back as a unified diff. |
| 6 | Every returned diff passes the quality gate in every access mode: deterministic checks, then an AI reviewer that compares the user's request, the agent's stated intent and the diff. Verdicts: `intended` → apply (auto) or show-and-ask (ask); `unclear` → ask with the reviewer's explanation; `suspicious` → refuse and flag. Application is `apply_patch` (all-or-nothing, write gates). |
| 7 | Default approval for dependency changes and diff application is ask; auto applies only on `intended` with no deterministic red flag. |
| 8 | First target: uv-managed Python projects; guru itself is the fixture. |
| 9 | Container escapes land in the Colima VM, not macOS; accepted for local development. |

## 1. Components and layers

**Domain (`guru/domain/`)**
- `sandbox.py` — `SandboxSpec` (image digest, project copy path, limits), `Verdict` (`intended|unclear|suspicious`, reasons), `gate_rules(diff, project) -> list[Flag]` (deterministic: paths inside project only, no noise dirs, size cap, secret scan, red-flag patterns: `subprocess`, `os.system`, `eval(`, `exec(`, `socket`, `ctypes`, base64 blobs > 200 chars, `@pytest.mark.skip`, deleted `assert`s, changed CI/config files), `decide(flags, review) -> Verdict`.
- `deps.py` — `DependencyRequest`, lockfile diff summary (packages added/removed/changed), rule "an install is only a lockfile change".
- reuse: `procs.py` runner interface, `patch.py` apply, `policy.py` scanner, `decisions.py` (the AI review is a decision point `gate` with an LLM judge; rows carry request, intent, verdict, reasons).

**Repository (`guru/repositories/`)**
- `sandbox_images.py` — per-project image records under `~/.guru/sandbox/<project>/` (Dockerfile guru generated, lockfile sha, image digest, built_at); `net_events` and `sandbox_events` streams in the JSONL ledger.
- `settings.py` — `[sandbox]` table: `runtime = "docker"`, `base_image = "python:3.12-slim@sha256:…"`, `cpus`, `memory_mb`, `pids`, `timeout_s`, `proxy_image`, `enabled = false` (off unless a project opts in via `.guru/sandbox.toml`).

**Endpoint**
- `guru/sandbox/colima.py` — the runtime: `build(spec)` (fixed argv `docker build` with the generated Dockerfile, on the internal network with proxy env), `run(spec, argv)` (`docker run --rm --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp --pids-limit N --memory M --cpus C -v <copy>:/work -w /work <image> argv…` with a wall-clock kill), `diff(spec) -> str` (`git diff` inside the copy), `proxy_up(allowlist)` / `proxy_down()` (tinyproxy or squid container with a generated allow-list config; access log tailed into `net_events`).
- `guru/sandbox/gate.py` — runs the deterministic rules, asks the `gate` judge (Sonnet by default via the ladder, fixed question set), records the decision row, returns the Verdict; on `intended` calls `apply_patch`; on `unclear` presents verdict + diff via the asker; on `suspicious` refuses and prints the reasons.
- Tools (registry): `sandbox_run(argv: list)` — fixed argv passed to the container (argv[0] must be in `{python, pytest, uv, ruff, mypy, flake8, make}`; shells refused as everywhere), `sandbox_python(code)` (writes code to a temp file in the copy and runs `python file` — arbitrary code is allowed INSIDE the sandbox), `request_dependency(name, constraint)`, `sandbox_diff()` (digest of the copy's changes), `sandbox_submit(intent)` (triggers the gate with the agent's stated intent; the only way changes reach the real tree). All gated by the tools policy; `sandbox_*` disabled unless the project has a sandbox spec.

## 2. Data flow

1. User opens a project with `.guru/sandbox.toml` (or runs `/sandbox init`). Guru generates the Dockerfile from `pyproject.toml` + `uv.lock`, asks (once) to allow `pypi.org` and `files.pythonhosted.org` if absent from the domain allow-list, starts the proxy on the internal network, builds the image, records it. Provision reruns only when the lockfile sha changes.
2. A worker calls `sandbox_run`/`sandbox_python`; guru copies the project (respecting noise dirs) into the sandbox work dir on first use per task, runs the container with `--network none`, returns a digest (same shape as `run_tests`: summary, first failures, `detail`), records `sandbox_events` (argv, seconds, exit, bytes).
3. The worker edits inside the copy via the container, verifies with `sandbox_run(['pytest', …])`, and calls `sandbox_submit(intent="…")`.
4. The gate: `sandbox_diff` → deterministic flags → `gate` judge with (user request, task text, agent intent, diff) → Verdict → apply / ask / refuse. Rows: decision `gate`, `sandbox_events` submit, `tool_events` for the patch.
5. `request_dependency` → recorded → ask → `uv add` in the provision container through the proxy → lockfile diff shown → rebuild → next `sandbox_run` uses the new image.

## 3. Threat model (what is and is not covered)

- Exfiltration during execution: impossible (no network). During provisioning: only allow-listed hosts over the proxy, all logged; the internal network still reaches the Colima VM's own gateway-address listeners (accepted, see decision 2), never published container ports or the LAN.
- Copy hygiene does not depend on routing: the working copy is the `git ls-files` positive list (gitignored files never enter) minus noise dirs, `.env*` and files the project secret scanner flags (bound by the endpoint when routing bound none).
- Host tampering: the sandbox sees only its copy; the real tree changes only through the gate + `apply_patch`.
- Secrets: no env, no home, no agent, no credentials in the container; the image is built from the copy, which is scanned with the secret scanner before build (a `.env` in the project is excluded from the copy).
- Supply chain: identical to installing the project's own lockfile; pinned base image digest; installs only via approved lockfile changes. uv itself is version-pinned via pip (not hash-pinned: the multi-arch base selects different platform wheels; a digest-pinned `COPY --from=ghcr.io/astral-sh/uv` binary is the planned fix). The proxy image is a pinned third-party digest (decision 3).
- Resource abuse: cpus/memory/pids limits, `--rm`, wall-clock kill, read-only root.
- Kernel escape: lands in the Colima VM. Accepted.
- Gate failure modes: a reviewer fooled by an obfuscated diff — mitigated by the deterministic red-flag rules running first and the review loop measuring the gate's verdicts against outcomes.

## 4. Implementation chunks

**S1 Colima runtime + spec + settings.** `sandbox/colima.py` (build/run/diff with fixed argv through `procs.run`; guru's own docker argv is assembled, never model text), `SandboxSpec`, `[sandbox]` settings, `.guru/sandbox.toml`, Dockerfile generation from `uv.lock`, image records. Tests: argv assertions with a fake `procs.run`; Dockerfile generation from a fixture lockfile; a real build+run integration test marked `sandbox` and skipped unless `docker info` succeeds against Colima.
**S2 Proxy + provisioning.** Internal network create/destroy, proxy container with generated allow-list from `config.ALLOWED_DOMAINS`, `net_events` tailing, `request_dependency` → approval → `uv add` → rebuild. Tests: allow-list generation; refusal of a non-listed host (integration, skipped without Colima); lockfile diff summary.
**S3 Verbs + gate.** `sandbox_run`, `sandbox_python`, `sandbox_diff`, `sandbox_submit`; `gate_rules`, the `gate` judge question set (fixed), Verdict handling in the three modes, `/sandbox status`. Tests: rules on crafted diffs (red flags, noise dirs, size), verdict mapping, submit applies via apply_patch on `intended` and asks on `unclear` (fake judge), refuses on `suspicious`.
**S4 Evals + docs.** Cases: `sandbox-fix-and-submit` (cli-tool: fix the failing test inside the sandbox, submit with intent, gate intended, fixture tests pass on the real copy), `sandbox-unrelated-change` (a planted worker that also edits an unrelated file → gate unclear/suspicious, nothing applied), `sandbox-dependency-request` (asks for a package → request recorded, no install, image unchanged). README "Sandbox" section; state-ownership; the eval report gains gate verdict counts.

Order: S1 → S2 → S3 → S4, each committed after green gates, security review after S1+S2 and after S3.

## 5. Notes from S2 (2026-09-24)

- **Proxy image**: `docker.io/kalaksi/tinyproxy` (tinyproxy 1.11.3 on Alpine, 6.9 MB, unprivileged), pinned by its multi-arch index digest in `settings.DEFAULT_PROXY_IMAGE`. `ubuntu/squid` was rejected for size; the tinyproxy project publishes no official image (`ghcr.io/tinyproxy/tinyproxy` denies).
- **"CONNECT 443 only, no plain HTTP"** is expressed with `FilterURLs On` + `FilterType ere` + `FilterDefaultDeny Yes` and a filter file of anchored `^host:443$` patterns: the request-URI of a `CONNECT` is exactly `host:port`, so a `http://host/...` URL or another port never matches. `ConnectPort 443` is belt and braces. `Allow <internal subnet>` keeps other bridge containers from using the proxy. Verified live: allowed CONNECT 200, other host refused, plain HTTP 403, other port refused, no route without the proxy.
- **BuildKit cannot attach a build to a user-defined network** (`network mode "x" not supported by buildkit`). `colima.build` therefore runs the classic builder (`DOCKER_BUILDKIT=0`) whenever the network is not `none`/`default`/`host`. The classic builder is deprecated in Docker 28 but present; the fallback if it is removed is `docker run` on the internal network + `docker commit`, behind the same `build()` signature.
- The proxy variables are passed as `--build-arg` only; Docker predefines these names, so they reach every `RUN` without `ARG` lines and stay out of `docker history` (declaring them would record the values). Verified in the integration test.
- `net_events` rows are aggregated per (host, port, method, allowed, reason) with a `count` and a `phase` (`build`, `uv add`); tinyproxy logs no byte counts.
- Per-project state (image tag, `~/.guru/sandbox/<key>/`) is keyed `<basename>-<sha8 of resolved path>`; network and proxy names carry a per-session suffix, so sessions never share or remove each other's provisioning resources, and a reused network is accepted only if `docker network inspect` says it is internal.
- `uv add` runs in the sandbox image on the internal network with `UV_OFFLINE=0` overriding the image's `UV_OFFLINE=1`, `--no-sync`, and `UV_CACHE_DIR` on the tmpfs; only `pyproject.toml`/`uv.lock` of the copy are read back, and they reach the real tree through `apply_patch`.
