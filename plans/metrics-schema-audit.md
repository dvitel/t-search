# Metrics Schema Audit

## Goal
Document every metric key emitted via `add_metrics(...)` across the codebase, how they're structured in the `.jsonlist` output, and what `postprocess/main.py` expects. Goal: make it easy to know what to add/check when building new operators.

## Status
DONE

## Progress
- Ran 3 parallel fork agents: add_metrics grep, solver/evaluator get_iter_metrics, postprocessor
- Wrote full schema into CLAUDE.md: top-level scalars, per-gen lists, per-operator sub-dicts, accumulation rules, postprocessor access patterns, how-to guide for new operators
- Key gotcha discovered: all operator add_metrics calls (no scope=) land in TOP-LEVEL dict — success/fail/repr from different operators pile into same counter

## Open questions
