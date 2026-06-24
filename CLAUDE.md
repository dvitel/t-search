# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based symbolic regression solver using genetic programming with semantic search. The solver (`GPSolver`) is scikit-learn compatible (`fit`/`predict`) and configurable via JSON pipeline configs. The core idea is **semantic genetic programming**: terms are evaluated as semantic vectors and stored/searched in vector spaces, enabling geometry-aware crossover and mutation.

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml --prefix ./t-search-env

# Install package in editable mode
./t-search-env/bin/pip install -e .
```

## Running

```bash
# Run a benchmark via CLI
./t-search-env/bin/python -m t_search \
    --dataset r_1 \
    --config ./configs/point_optim.json \
    --output ./data/results.jsonlist \
    --device cuda \
    --dtype float32 \
    --seed 42

# Run multiple datasets via batch scripts in local/
bash local/koza1.sh
```

## Testing

```bash
./t-search-env/bin/pytest           # all tests
./t-search-env/bin/pytest tests/test_t1.py  # single test file
```

## Architecture

### Core pipeline flow

`GPSolver.fit()` → `on_start()` (instantiates all services from JSON config via DI) → `_check_trivial()` → `_loop()` (runs operators each generation).

The solver is **service-oriented**: every component (syntax, evaluator, fitness, semantics, operators) is registered as a named service. Config JSON declares service types with `{"type": "ClassName", "params": {...}}` and `{"!ref": "service_name"}` for cross-references. `solver_utils.register_services` resolves the dependency graph, injecting constructor params from either the config or a shared `injection_context` dict (populated with runtime values like `ops`, `free_vars`, `target`, `var_bindings`, etc.).

### Key modules

- **`t_search/syntax/`** — Term representation. `Term` is the base; subtypes are `Op` (function node), `Variable` (leaf), `Value` (constant leaf), `OptimPoint` (learnable constant marker). `Syntax` (in `syntax.py`) is a **term cache/factory**: allocating the same op+args always returns the same object (interning), enabling identity-based equality. It also manages constraint checking (depth, op counts, forbidden patterns via unification).

- **`t_search/evaluators/`** — `Evaluator` computes semantic outputs (torch tensors) for terms over the training data. `Semantics` stores term→vector mappings; `Fitness` computes NMSE/MSE/L1 loss; `ConstOptimizer` runs gradient-based optimization of `Value` constants.

- **`t_search/spatial/`** — Vector index structures used by semantic operators: `VectorStorage` (cosine/L2 approximate search), `BinStorage`, `GridStorage`, `RTreeStorage`, `SpearmanStorage`. Used to find semantically similar/dissimilar terms efficiently.

- **`t_search/operators/`** — Evolutionary operators, organized by algorithm family:
  - `classic/` — Standard GP: `RHH` (Ramped Half-and-Half init), `RPM`/`RPX` (random point mutation/crossover), `TournamentSelection`, `MuLambdaSurvivorSelection`, `Up2D` (library up to depth 2)
  - `competent/` — Competent GP (Goldberg-style): `DesiredSemanticLib` + `CompetentMutation`/`CompetentCrossover`/`CompetentSelection`
  - `geometric/` — Geometric semantic operators (Moraglio-style)
  - `semantic/` — Semantic-aware operators
  - `optim/` — Constant optimization-based mutation (`ConstOptimMutation`)
  - `llm/` — LLM-based mutation operators (requires `jinja2`, `openai`, `google-genai`)
  - `mcts/` — Monte Carlo Tree Search operators
  - `moo/` — Multi-objective operators

- **`t_search/datasets/`** — Benchmark problem definitions (`Benchmark` class wrapping a callable + sampling config). Datasets (Koza, Nguyen, Keijzer, Vladislavleva, Korns, Pagie, etc.) are registered as module-level attributes and referenced by name via `--dataset`.

- **`postprocess/main.py`** — Post-processing and visualization of result `.jsonlist` files (tables, plots for papers).

### Config system

`configs/` contains JSON pipeline configurations. `configs/_full.json` is the canonical template showing all options. `configs/README.md` documents all parameters. Key config keys:

- `service_definitions` — dict of named services with `type`, optional `module`, and `params`
- `init_service_name`, `operator_service_names`, `evaluator_service_name`, `syntax_service_name`, `fitness_service_name`, `semantics_service_name` — wire services into solver roles
- `ops` — list of allowed symbols (e.g. `["add", "mul", "sin", "cos"]`)
- `max_gen`, `max_root_evals`, `max_evals` — termination criteria

### Output format

Results are written as newline-delimited JSON (`*.jsonlist`), one record per run, including config name, dataset, seed, metrics per generation, and final best term.

### Adding a new operator

1. Implement a class in the appropriate `t_search/operators/<family>/` submodule inheriting from `Operator` (or `Initialization` for init operators).
2. Constructor params are auto-injected from `injection_context` by name — declare them normally. Use `{"!ref": "service_name"}` in config to inject other services.
3. Reference the operator in a config JSON with `"module": "t_search.operators.<family>"` and `"type": "YourClass"`.

## Development Workflow

All development follows a goal-driven loop. Goals may have subgoals and phases.

### Phase 0 — Goal intake
When a goal is given, create a plan file in `plans/` named after the goal (e.g. `plans/goal-slug.md`). The file starts with the stated goal and vision. Update it throughout — it acts as a living scratchpad: progress notes, design decisions, indices of relevant files/classes, open questions. You have full edit rights on files in `plans/`.

Plan file structure (use as a template):
```
# <Goal title>

## Goal
<Stated goal and vision from the user>

## Status
EXPLORING | PLANNING | IMPLEMENTING | TESTING | DONE | ABANDONED

## Plan
<Filled in after step 2 approval>

## Progress
<Running notes — files touched, decisions made, blockers>

## Open questions
<Questions for the user before finalizing the plan>
```

Goals can nest or stack. When a subgoal or side goal is started mid-way through a parent goal, create its own plan file and add a note in the parent plan's Progress section pointing to it (e.g. "→ jumped to `plans/subgoal-slug.md`"). When returning to the parent, note that too ("← returned from `plans/subgoal-slug.md`"). If a goal is abandoned, set status to `ABANDONED` and add a brief reason in Progress.

### Phase 1 — Exploration and questions
Explore the codebase (spawn fork agents for parallel reads when the scope is wide — agent activity is visible in the console). After exploration, list any open design/clarification questions for the user before drafting the plan. Keep questions concrete and decision-oriented.

### Phase 2 — Plan presentation and approval
Present a concise implementation plan. The user will approve it or request changes; repeat phases 1–2 until approved. Approval also means "proceed to implementation" unless the user says otherwise. Record the approved plan in the plan file and set status to `IMPLEMENTING`.

### Phase 3 — Implementation
Implement the goal. Keep the plan file updated with progress. After implementation, suggest specific manual or automated tests to verify the solution. Prefer pytest tests in `tests/` when the logic is unit-testable; otherwise describe an exact CLI invocation to run.

### Phase 4 — Review and completion
The user may request further changes. When the user confirms the goal is accomplished, mark the plan file status as `DONE` and save a memory entry summarizing what was built and any non-obvious decisions made. The user will commit.

**On "out of scope":** only call out closely related functionality that is explicitly not being touched when there is a real risk of confusion — not as a blanket list.
