"""Functional evaluation suite: prompts run against frozen fixture repos.

Layers: ``cases`` (TOML case format) and ``checks`` (pure assertions) are
domain; ``runs`` (run files, compare, trajectory) is a repository; the
runner over the headless orchestrator is the endpoint. Design:
docs/plans/2026-09-23-routing-framework-design.md §8.
"""
