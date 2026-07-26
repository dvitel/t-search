# PointOptim Algorithm Improvement Vectors

## Goal
Document identified improvement directions for the `PointOptim` operator, informed by code analysis and algorithmic reasoning. Serves as a reference for future development goals. No implementation yet.

## Status
PLANNING

## Context
`PointOptim` (`t_search/operators/optim/point_optim.py`) is the most promising operator in the codebase. Core idea: gradient-optimize a "hole" (OptimPoint placeholder) to find the ideal semantic output for a position, then search the library for the syntactic expression that best approximates it (with linear scaling k·t + b). Some parts of the current code are stubs or leftovers from earlier design iterations (e.g. LossBasedContinuation insertion is commented out, some backtrack paths are partially connected).

---

## Improvement Directions

### 1. Multi-position optimization (PRIORITY — next goal)

**Idea (user's):** Instead of optimizing one position at a time, identify several positions in the same tree that could benefit from the *same subtree* with different linear scalings. Use gradient measurements to estimate which positions are good candidates for such substitution. Find the common subterm and scale it independently for each selected position, then rebuild the tree in one move.

**Why it matters:** Single-position optimization is locally optimal but misses correlated positions. If two positions share structure (e.g. both want `sin(x)` but scaled differently), optimizing them jointly via a shared basis is more powerful than sequential single-hole fills.

**Connection to epistasis:** In GP/GA theory, epistasis = interaction between gene positions (changing one affects the fitness contribution of another). Multi-position optimization directly addresses this: when positions are epistatically linked, one-at-a-time optimization diverges or takes many steps. A joint fill can solve it in one move. Finding a benchmark problem where single-point mutation diverges but multi-point succeeds would be a strong empirical justification.

**Algorithm sketch:**
- For a given term, run `get_all_grads` to rank positions by gradient magnitude
- Select top-K positions (K=2..4) as candidate holes
- Replace all K with distinct `OptimPoint(0)`, `OptimPoint(1)`, ... placeholders
- Run joint multi-dimensional LBFGS to find the K ideal vectors simultaneously
- For each position, search library independently for best (k, subterm, b) matching its optim vector
- Optionally: constrain search to find the *same* stripped subterm for all positions (different k, b per position) — this is the "common subterm" variant

**Open questions:**
- How to handle depth budget across K holes simultaneously?
- Does forcing a common subterm meaningfully help, or is independent per-position search sufficient?
- Which gradient ranking to use: magnitude, variance, or something else?

---

### 2. Position-local range propagation

Use `backward_desired` (already in `competent/utils.py`) to analytically propagate the target through parent operators before running multi-start optimization. Tightens the optimization range and biases initial starts toward useful regions. For nonlinear ops (`sin`, `exp`), fall back to global range.

Currently: ranges are always `[min_y − δ, max_y + δ]` regardless of what operators sit above the hole.

---

### 3. Multi-term filling (sparse linear combination)

Instead of `k·t + b` (single library term + affine), solve a small OLS/LASSO over the top-N library terms:
`Σ kᵢ·tᵢ + b`. Useful when the optim vector lies between two library terms semantically. `LincombMixin` already has infrastructure for this.

---

### 4. Adaptive library ("hall of fame")

When a fill succeeds (`num_better_fills` increments), add `context.filling` to a persistent growing library. Terms that worked once are likely useful again in different positions. Cheap to implement; improves later generations when the static Up2D library is exhausted.

---

### 5. Tabu decay

Replace monotone `tabu_set: set[Term]` with `tabu_set: dict[Term, int]` (term → generation added). Un-tabu a term if `current_gen − tabu_gen > K`. Allows revisiting positions after surrounding tree structure has changed enough. K could adapt based on population average depth.

---

### 6. Exploit dist–loss correlation adaptively

Currently Pearson/Spearman correlation between semantic distance and loss improvement is tracked at finalizer time (diagnostic only). If correlation is consistently low on a given problem, distance-based fill ordering is essentially random — signal to switch to loss-based or try a different normalizer. Could be a mid-run adaptation rather than post-hoc diagnostic.

---

### 7. Hybrid library search + guided random growth

After library search, if the best library match is still far from the optim vector (distance > threshold), seed a random tree grow using the optim vector as a semantic target (grow toward it via `SemanticallyDrivenInitialization`-style filtering). Library search and guided growth complement each other.

---

## Notes on current code state

- `LossBasedContinuation` insertion is commented out in `best_by_metric == "loss"` branch — continuation mechanism is partially built
- Several backtrack paths are stubs or partially connected — intentional leftovers from previous design iterations
- `tried_optim_terms` / `tabu_set` grow unboundedly — memory concern on long runs
- `with_pop_terms=True` dedup is O(n²) in query size — avoid with large populations

## Progress
- Code analysis complete
- Improvement directions documented from exploration session
- Multi-position optimization identified as next concrete goal (not started)
