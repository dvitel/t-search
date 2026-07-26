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

Results are written as newline-delimited JSON (`*.jsonlist`), one record per run. See **Metrics Schema** section below for full key reference.

### Adding a new operator

1. Implement a class in the appropriate `t_search/operators/<family>/` submodule inheriting from `Operator` (or `Initialization` for init operators).
2. Constructor params are auto-injected from `injection_context` by name — declare them normally. Use `{"!ref": "service_name"}` in config to inject other services.
3. Reference the operator in a config JSON with `"module": "t_search.operators.<family>"` and `"type": "YourClass"`.

---

## Metrics Schema

Every run appends one JSON line to the output `.jsonlist` file. Keys are structured as follows.

### `add_metrics` accumulation rules

```python
add_metrics(metrics_dict, key=value)
# First write  → metrics[key] = value
# List value   → metrics[key].extend(value)   (per-gen arrays grow this way)
# Scalar value → metrics[key] += value         (counts accumulate)
```

**Scope caveat:** `step_time` / `total_time` per operator go into a **sub-dict** keyed by operator service name (e.g. `record["mutation"]`). All other `add_metrics` calls from operators go into the **top-level** dict — so keys like `success` / `fail` / `repr` from different mutation operators in the same pipeline all sum into the same top-level counter.

---

### Top-level scalars (always present)

| Key | Type | Description |
|---|---|---|
| `config_name` | str | Config file basename — primary grouping key |
| `dataset` | str | Benchmark name |
| `seed` | int | RNG seed |
| `config` | str | Full config path |
| `test_nmse` | float | NMSE on test set — primary result metric |
| `test_pred_num_invalid` | int | Non-finite predictions on test set; **postprocessor filters records where this ≥ 10** |
| `test_num_samples` | int | Test set size |
| `train_num_samples` | int | Train set size |
| `gen` | int | Final generation reached |
| `final_time` | int (ms) | Total wall time |
| `status` | str | `MAX_GEN` \| `MAX_EVAL` \| `MAX_ROOT_EVAL` \| `SOLVED` \| `DEADEND` |
| `best_term` | str | LISP string of best formula |
| `best_fitness` | float | Best train NMSE |
| `best_term_depth` | int | Depth of best term |
| `best_term_size` | int | Node count of best term |
| `init_time` | int (ms) | Init operator time |
| `init_eval_time` | int (ms) | Initial eval time |
| `total_eval_time` | int (ms) | Cumulative eval time |
| `evals` | int | Total term evaluations |
| `root_evals` | int | Root-level evaluations |
| `optim_evals` | int | Evaluations inside const optimizers |
| `eval_calls` | int | `eval()` invocation count |
| `eval_cache_hits` | int | |
| `eval_cache_miss` | int | |
| `loss_trace` | list[float] | Best NMSE sampled every `loss_each_n` root evals (up to 200 entries) |
| `loss_each_n` | int | Sampling interval for `loss_trace` |
| `consts_used` | int | Constants allocated from tape |
| `invalid_terms` | int | Non-finite terms at finalizer |
| `syntax_hit` | int | Syntax cache hits (summed) |
| `syntax_miss` | int | Syntax cache misses (summed) |
| `backtracks` | int | Tree-grow backtracks (summed) |
| `gen_fails` | int | Tree-grow failures (summed) |
| `success` | int | Successful mutations (summed across all `TermMutation` operators) |
| `fail` | int | Failed mutations (summed) |
| `repr` | int | Unchanged terms returned (summed) |

---

### Per-generation list keys (length = `gen`)

| Key | Description |
|---|---|
| `iter_time` | Wall ms per generation |
| `iter_fitness` | Best NMSE per generation |
| `iter_term_size` | Node count of best term per gen |
| `iter_term_depth` | Depth of best term per gen |
| `iter_num_syntax` | Syntax cache size per gen |
| `iter_num_consts` | Constants allocated per gen |
| `iter_evals` | Cumulative total evals at gen end |
| `iter_root_evals` | Cumulative root evals |
| `iter_optim_evals` | Cumulative optim evals |
| `iter_evals_simple` | Cumulative leaf-node evals |
| `iter_eval_calls` | Cumulative `eval()` calls |
| `iter_eval_cache_hits` | Cumulative cache hits |
| `iter_eval_cache_miss` | Cumulative cache misses |
| `iter_num_semantics` | Unique semantic vectors stored per gen |
| `iter_num_sem_terms` | Total terms in semantic storage per gen |
| `iter_num_invalid_terms` | Invalid terms per gen |

---

### Per-operator sub-dicts (keyed by service name)

Accessed as `record["<service_name>"]`. Every operator has:

| Key | Description |
|---|---|
| `step_time` | list[int] ms — wall time per gen call |
| `total_time` | int ms — cumulative |

Plus operator-specific keys within the sub-dict:

**`PointOptim` scope:**
`num_better_fills`, `num_total_fills`, `num_holes_created`, `num_terms_optimized`, `tried_optim_terms`, `tried_optim_terms_hit`, `tabu_positions` (int); `pearson_dist_loss`, `pearson_dist_loss_p_value`, `spearman_dist_loss`, `spearman_dist_loss_p_value` (float | None)

**`evaluator` scope** (in addition to `step_time`/`total_time`):
`evals`, `root_evals`, `optim_evals`, `eval_calls`, `eval_cache_hits`, `eval_cache_miss`, `loss_trace`, `loss_each_n`

---

### Operator-specific top-level keys

**`LLMMutation` / `LLMSelection`:** `render_error`, `llm_error`, `llm_syn_invalid`, `llm_constr_invalid`, `good_mutations`, `bad_mutations`, `neutral_mutations`, `invalid_index`, plus raw LLM API usage dict (`input_tokens`, `output_tokens`).

**`ReduceConsts`:** `sem_cache_miss`, `eval_time`

**`Reduce` (Sympy):** `simplify_time`, `validity_check_time`

**`CompetentInitialization`:** `target_inside_hull` list[int]

---

### Postprocessor access

`postprocess/main.py` reads records with nested key access via `.` separator (e.g. `"evaluator.loss_trace"` → `record["evaluator"]["loss_trace"]`). It only uses top-level scalars for tables (`test_nmse`, `best_fitness`, `final_time`, `best_term_size`) and `evaluator.loss_trace` for convergence plots. All per-generation lists and operator sub-dicts are ignored by the postprocessor but available for custom analysis.

---

### Adding metrics in a new operator

```python
# Inside __call__ or mutate_term / mutate_position:
self.add_metrics(my_counter=count, my_timing=[elapsed_ms])
# Scalar → accumulates via +
# List   → extended each call (use for per-gen time series)

# For keys that should live in the operator's own sub-dict,
# they must be emitted via scope= in the solver loop (automatic for step_time/total_time).
# Direct add_metrics calls from operators always land in the top-level dict.
```

---

## Operator & Component Index

### Base classes / mixins (operators)

| Class | File | Role |
|---|---|---|
| `Operator` | `operators/operator.py` | Base `__call__(pop) -> pop` |
| `Initialization` | `operators/operator.py` | Base `__call__() -> pop` |
| `TermMutation` | `operators/mutation.py` | Single-term transform; wraps `mutate_term(t) -> t` |
| `PositionMutation` | `operators/mutation.py` | Position-level transform; wraps `mutate_position(t, pos, ...) -> t` |
| `TermCrossover` | `operators/crossover.py` | Pairwise transform; wraps `crossover_terms(t1, t2) -> t` |
| `Selection` | `operators/selection.py` | Pop → selected subset |
| `SurvivorSelection` | `operators/survivor_selection.py` | (parents, children) → next gen |
| `LincombMixin` | `operators/reduction.py` | Adds `build_lincomb` for constructing linear combinations of terms |
| `GenListener` | `operators/listeners.py` | `on_gen_start` / `on_gen_end` hooks |

---

### Initialization operators

| Class | File | Description |
|---|---|---|
| `RHH` | `classic/rhh.py` | Ramped Half-and-Half: random depth in `[min_depth, max_depth]`, toggles full/grow; retries to avoid syntactic duplicates. Params: `size`, `min_depth`, `max_depth`, `grow_proba`, `syntactic_duplicate_retries`. |
| `RHHCached` | `classic/rhh.py` | Fills population from the global syntax cache rather than growing new trees. Params: `vars`, `syntax` + all `RHH` params. |
| `Up2D` | `classic/up2d.py` | Enumerates all valid trees up to `depth` (optionally with const=1). Deterministic library seed. Params: `depth`, `const_1`, `max_size`. |
| `SemanticallyDrivenInitialization` | `semantic/initialization.py` | Beadle & Johnson 2009a. Builds population by combining existing programs; rejects semantically duplicate outputs. Params: `size`, `max_depth`, `atol`, `rtol`, `max_l2`. |
| `CompetentInitialization` | `competent/initialization.py` | Experimental: convex-hull over semantic vectors; currently falls back to `SemanticallyDrivenInitialization`. |

---

### Selection operators

| Class | File | Description |
|---|---|---|
| `TS` | `classic/ts.py` | Tournament selection (GPU-batched fitness tensor). Params: `tournament_size`, `add_pop`. |
| `CompetentSelection` | `competent/selection.py` | Tournament for fitness winner + partner chosen to maximize `target_dist / parent_dist * |Δfitness|`. Promotes semantic diversity. Params: `selection_size`, `tournament_size`. |
| `SemanticTournamentSelection` | `semantic/selection.py` | Galván-López 2013. Standard tournament, but second parent must be semantically distinct. Params: `rtol`, `atol`. |
| `FrontierSelection` | `optim/po_selection.py` | Draws candidates from `term_frontier` (best PointOptim targets) ∪ population. Optionally expands to best subterm. Params: `with_subterms`. |
| `Lexicase` | `moo/lexicase.py` | Lexicase: shuffled test cases, per-test Pareto front elimination. Params: `nan_error`. |
| `LLMSelection` | `llm/selection.py` | LLM picks tournament winner from rendered LISP terms + fitness values. Params: `llm`, `tournament_size`, `prompt_template`. |

---

### Mutation operators

| Class | File | Description |
|---|---|---|
| `RPM` | `classic/rpm.py` | Random Point Mutation: pick position, replace subtree with freshly grown tree. Params: `max_grow_depth`, `freq_skew`. |
| `CompetentMutation` | `competent/mutation.py` | Kraviec & Pawlak. Backpropagates desired semantics through parent, fills positions from `DesiredSemanticLib`. Params: `lib`, `identity_atol/rtol`, `rate`. |
| `SemanticGeometricMutation` | `geometric/mutation.py` | Moraglio 2012: `p' = p + r·(t1 − t2)`. Optimal r computed analytically, then perturbed by `alpha`. Params: `min/max_grow_depth`, `epsilon`, `alpha`. |
| `SemanticallyDrivenMutation` | `semantic/mutation.py` | Beadle & Johnson. RPM + reject if semantic distance outside `[min_d, max_d]`. |
| `ConstOptimMutation` | `optim/const_optim.py` | Gradient-based constant tuning (LBFGS) on the `n_best_terms` cheapest individuals. Params: `const_optimizer`, `n_best_terms`. |
| `PointOptim` | `optim/point_optim.py` | Hole-filling: inserts `OptimPoint` at each position, gradient-optimizes to find best scalar, then searches semantic lib for closest-match subterm. Supports tabu, backtracking, continuation queues. Params: `position_strategy`, `num_starts`, `num_lib_terms`, `num_conts`, `backtrack`. |
| `LLMMutation` | `llm/mutation.py` | LLM proposes a skeleton from hardest test cases + ICL exemplars; constants are then gradient-optimized. Params: `llm`, `prompt_template`, `num_last_mutations`, `max_num_tests`. |
| `BestSubterm` | `semantic/best_subterm.py` | Replaces each term with its best-fitness (or most-tests-winning) subterm. Params: `test_based`, `test_based_percent`. |
| `BestInner` | `misc/best_inner.py` | Replaces each term with its best-fitness inner subterm (cached). |

---

### Crossover operators

| Class | File | Description |
|---|---|---|
| `RPX` | `classic/rpx.py` | Random Point Crossover: swap a random subtree of one term into a random position of another. |
| `CompetentCrossover` | `competent/crossover.py` | Midpoint semantic target between two parents, backpropagated and filled from lib. Params: `lib`, `identity_atol/rtol`, `rate`. |
| `SemanticGeometricCrossover` | `geometric/crossover.py` | Moraglio 2012: `p' = ε·p1 + (1−ε)·p2` with analytically optimal ε. Params: `epsilon`, `alpha`. |

---

### Survivor selection

| Class | File | Description |
|---|---|---|
| `MuLambdaSurvivorSelection` | `classic/mu_lambda.py` | (μ+λ) with `combine=True` or (μ,λ) with `combine=False`. Optional elitism. Params: `mu`, `strict`, `combine`, `elitism`. |

---

### Post-processing / simplification

| Class | File | Description |
|---|---|---|
| `SReduce` | `semantic/reduce.py` | Replaces subterms with semantically equivalent simpler representatives from the cache. |
| `ReduceConsts` | `syntax/reduce_consts.py` | Replaces constant-output subtrees with a single `Value` leaf. |
| `Reduce` | `syntax/reduce.py` | Sympy algebraic simplifier. Also exports `to_sympy`, `from_sympy`, `sp_simplify` as standalone functions. |

---

### Utility operators

| Class | File | Description |
|---|---|---|
| `Dedupl` | `misc/dedupl.py` | Removes syntactically duplicate terms from the population (identity-based). |
| `Valid` | `misc/valid.py` | Filters population to syntactically + semantically valid terms only. |
| `Logging` | `misc/logging.py` | `GenListener` — logs generation events and per-term evaluations to file or stdout. |

---

### Libraries (shared state for competent/point-optim operators)

| Class | File | Description |
|---|---|---|
| `DesiredSemanticLib` | `competent/base.py` | Pre-evaluates seed terms (via `init_op`) at startup; stores their semantic vectors for competent mutation/crossover. Params: `init_op`. |

---

### MCTS (partial / in progress)

| Class | File | Description |
|---|---|---|
| `MCTreeNode` | `mcts/base.py` | Dataclass: term, producing operator, UCB Q-value, visit count, children, parent. |
| `MCTree` | `mcts/base.py` | Container with `pick_node`, `create_child`, `update_nodes` (backprop). |
| `MCTSInitialization` | `mcts/initialization.py` | Stub. |
| `MCTSSelection` | `mcts/selection.py` | Stub. |

---

### Spatial / Vector Index (`t_search/spatial/`)

All indices store semantic output vectors (one vector per term, length = number of training points).

| Class | File | Description |
|---|---|---|
| `VectorStorage` | `base.py` | Flat O(n) tensor store. `find_close`, `find_closest`, `find_in_range`. Params: `dims`, `capacity`, `rtol`, `atol`. |
| `SpatialIndex` | `base.py` | Extends `VectorStorage` with deduplicating insert (`find_unique` before alloc). Base for all concrete indices. |
| `BinIndex` | `bin.py` | Abstract: maps vectors to bin keys, routes queries to candidate bins, rebuilds on overflow. Subclass and implement `get_bin_index`. |
| `GridIndex` | `grid.py` | Uniform grid, adaptive bin splitting on rebuild (halves ε for most-imbalanced dims). Params: `epsilons`. |
| `InteractionIndex` | `inter.py` | Bins by binary proximity-to-target vector packed as int64 bitmask. Params: `target`. |
| `RCosIndex` | `cos.py` | Bins by (L2 norm, cosine distance to target). O(1) lookup. Broken with float16. Params: `target`. |
| `RTreeIndex` | `rtree.py` | R-Tree with recursive median split. Slower than grid in practice. |
| `SpearmanCorIndex` | `spearman.py` | Bins by Spearman correlation to target. Slow; needs float32 and large `max_children`. |

---

### Normalizers (`t_search/evaluators/term_spatial.py`)

Used by `PointOptim` and hole-based operators to normalize semantic vectors before distance computation.

| Class | Description |
|---|---|
| `IdentityNormalizer` | No-op. |
| `ZScoreNormalizer` | Z-score; flips sign if negatively correlated with target. |
| `ZRankNormalizer` | Z-score on ranks (Spearman-aligned geometry). |
| `GaussRankNormalizer` | Van der Waerden / probit transform on ranks; most outlier-robust. |

---

### Core Services (`t_search/evaluators/`)

| Class | File | Description |
|---|---|---|
| `Evaluator` | `evaluator.py` | Executes terms over training data, caches outputs, tracks `max_root_evals` / `max_evals` budgets, raises `EvSearchTermination` on budget hit. |
| `Fitness` | `fitness.py` | Stores NMSE (or MSE/L1) per term; tracks global best; raises `EvSearchTermination("SOLVED")` when fitness < `fitness_atol`. |
| `Semantics` | `semantics.py` | Facade over `TermVectorStorage`: resolves Variable→var_bindings, Value→scalar, Op→stored vector. |
| `TermVectorStorage` | `term_spatial.py` | Maps terms ↔ semantic IDs; wraps a `VectorStorage` index; handles invalid (non-finite) terms separately. |
| `HoleVectorStorage` | `term_spatial.py` | Like `TermVectorStorage` but keyed on `(Term, TermPos)` pairs — used by `PointOptim`. |
| `ConstOptimizer` | `const_optimizer.py` | LBFGS multi-start constant optimization. Replaces `Value` leaves with `OptimPoint`, runs `optimize_consts`, caches by skeleton. Params: `num_starts`, `max_evals`, `lr`. |

Key standalone functions in `evaluators/optimization.py`: `optimize_consts` (sequential multi-start LBFGS), `get_all_grads` (gradient w.r.t. every subterm position — used by gradient-guided operators), `optimize_par` (parallel batched starts).

---

### Benchmark Datasets (`t_search/datasets/algebra.py`)

All datasets are module-level `Benchmark` instances, referenced by name via `--dataset`.

| Suite | Names | Notes |
|---|---|---|
| Koza | `koza_1..3` | Classic 1-var polynomials |
| R | `r_1, r_2` | Rational functions |
| Nguyen | `nguyen_1..12` | 1–2 var, mix of poly/trig/exp |
| Pagie | `pagie_1, pagie_2` | 2-var rational; dense grid sampling |
| Korns | `korns_1..15` | 5-var; random points in [-50,50] |
| Keijzer | `keijzer_1..15` | 1–2 var; various ranges |
| Vladislavleva | `vladislavleva_1..8` | 1–5 var; harder targets |
| Test | `test_0` | Quick sanity-check problem |

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
