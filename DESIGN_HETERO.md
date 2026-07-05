# BEFGC-Hetero: Design Decisions Beyond the Paper

The BEFGC paper (`GW_coarsening.pdf`) is **homogeneous only**: a single graph
`G=(V,A,X)`, one coarsening matrix `C`, one adjacency `A`, one feature matrix
`X`. It never addresses multiple node/edge types. Everything below is an
addition or a decision we made for the heterogeneous setting.

Tags: **[BEFGC]** from the coarsening paper · **[AH-UGC]** adopted from the
paper we benchmark against · **[OURS]** our own formulation/derivation ·
**[PRACTICAL]** numerical/engineering choice.

---

## 1. Coarsening structure (type isolation)

- **[AH-UGC] Type-isolated coarsening.** Only nodes of the same type may merge
  into a supernode. Every supernode has a well-defined type. This is required
  for the coarsened graph to remain a valid heterogeneous graph and for the
  downstream hetero GNN to consume it.
- **[OURS] Realized as a block-diagonal `C` (single matrix + type mask).**
  Instead of a list of per-type matrices `C^(t)`, we keep ONE `C` of size
  `N x M` (N = all nodes, M = all supernodes) and force `C_ia = 0` whenever
  `type(i) != type(a)`. Mathematically identical to per-type matrices, but lets
  us reuse the homogeneous mirror-descent machinery almost verbatim.
- **[OURS] The mask is self-maintaining under the mirror update.** The
  multiplicative step `C_ia <- C_ia · exp(-η G_ia)` keeps zeros at zero
  (`0·exp(.)=0`), so the block structure never has to be re-applied after
  initialization. (In the FGC hetero baseline the mask is re-multiplied every
  step; we don't need to.)

## 2. Structural term — the core novelty (Option A)

- **[OURS] Cross-type bipartite Gromov-Wasserstein term.** BEFGC's structural
  term compares `A` vs coarsened `Ac` through the coupling. We compute it on the
  FULL global block adjacency `A_global`, whose off-diagonal blocks ARE the
  bipartite cross-type relations (movie-actor, movie-director, ...). With a
  block-diagonal `C`, `Ac = Cᵀ A_global C` automatically produces the correct
  per-edge-type coarsened blocks, so the GW discrepancy couples different node
  types directly through the transport plan. This is what makes our method
  *optimization-based cross-type coarsening* — AH-UGC does this by hashing, no
  objective.
- **[OURS] Transpose correctness.** The GW cross term is `A P Acᵀ`, not `A P Ac`.
  On undirected homogeneous graphs `Ac` is symmetric and the two coincide (which
  hid the bug on Cora), but across types `Ac` is rectangular/asymmetric, so the
  `.t()` is mandatory here. Fixed before the hetero extension.
- **[OURS] Why not metapaths (the rejected Option B).** The alternative was to
  build within-type similarity graphs via metapaths (e.g. movie-actor-movie) and
  run near-homogeneous BEFGC per type. Rejected: it discards the direct
  cross-type coupling and adds an arbitrary metapath-choice hyperparameter.

## 3. Feature term

- **[OURS] Computed per type.** Node types can have different feature dimensions
  (e.g. DBLP: Author=334, Paper=4231, Term=50), so a single `‖X_i − Xc_a‖²` is
  undefined across types. We compute the feature cost block-by-block and place it
  on the block diagonal; off-diagonal (cross-type) entries are left 0 because
  they are masked out of `C` anyway.
- **[PRACTICAL] L2 feature normalization (inherited from the homogeneous side).**
  Not in the paper; added so the feature term doesn't dominate and destabilize
  the mirror step. Applied per type.

## 4. Balance term

- **[OURS] Per-type balance target.** BEFGC balances supernode mass toward global
  uniform `1/m`. In the hetero case that is wrong: type `t` holds total mass
  `n_t/N` which should spread over its `m_t` supernodes, so the balanced target
  is `ν̄^(t)_a = (n_t/N)/m_t`. Balance is enforced *within each type*, not across
  types (mixing types is already forbidden by the mask).

## 5. Node mass, init, sizes

- **[PRACTICAL] Node mass `μ` = global uniform `1/N`.** Paper allows
  degree-/feature-aware masses; we use uniform as the paper's default.
- **[OURS] Block-diagonal row-stochastic init.** Each type block is initialized
  as a random positive row-stochastic matrix (same recipe as homogeneous), all
  cross-type entries zero, so the type mask holds from iteration 0.
- **[AH-UGC/OURS] Per-type target sizes `m_t = round(r · n_t)`.** One global
  coarsening ratio `r` applied per type (AH-UGC also allows different ratios per
  type; we use the same `r` for all types for now). Benchmarks at r=0.3 to match
  AH-UGC Table 4.

## 6. Coarsened graph construction (output)

- **[AH-UGC] Per-edge-type coarsened adjacency** `Ac^(e) = C^(t1)ᵀ A^(e) C^(t2)`
  — read off as the corresponding block of our global `Ac`.
- **[AH-UGC] Per-type coarsened features** `Xc^(t)` = barycentric (mass-weighted)
  aggregate of member features.
- **[AH-UGC] Labels by majority vote, target type only** (only `movie` carries
  labels in IMDB).

## 7. Evaluation protocol

- **[AH-UGC] Backbones:** HeteroSGC, HeteroGCN, HeteroGCN2 (the exact models in
  AH-UGC Table 4). Ported locally in `models_hetero.py`.
- **[AH-UGC] Train-on-coarse / test-on-original:** train the hetero GNN on the
  coarsened graph with majority-vote labels, then evaluate on the ORIGINAL
  graph's target-type test nodes.
- **[PRACTICAL] Empty supernodes masked out of the training loss;** `ν` clamped
  away from 0 in every division so an unused supernode can't produce NaNs.
- **[PRACTICAL] Both edge directions materialized** for each relation so hetero
  message passing can reach the target type.

## 8. Optimization / numerics

- **[PRACTICAL] Lower step size `η` for stability.** At the hetero scale
  (thousands of supernodes) `η≈0.3` with strong `β` makes the mirror step
  overshoot and the objective DIVERGE (empirically observed: J climbing,
  collapse worsening to ~60% empty). The hetero sweep searches `η ∈ [0.02, 0.25]`
  and the stable region is `low α (feature-heavy) + moderate β + η≈0.04–0.13`.
- **[PRACTICAL] Divergence tracking.** The sweep flags any trial whose final `J`
  exceeds its initial `J` and excludes it from "best", so an unstable config
  can't be reported as a win.
- **[BEFGC] Log-domain stabilized multiplicative update** (logsumexp per row) —
  carried over from the homogeneous implementation.
- **[PRACTICAL] Sparse global adjacency.** `A_global` kept as a sparse tensor;
  `A∘A` handled on stored values (correct for weighted graphs, and `= A` for 0/1
  adjacency) to hit the paper's complexity instead of dense `N×N`.

## 9. Still-open / not-yet-decided

- Different coarsening ratio per type (AH-UGC supports it; we use one `r`).
- Whether to pass coarsened edge WEIGHTS into the hetero GNN (currently the
  hetero backbones use unweighted coarse edges; the homogeneous eval keeps
  weights — a deliberate asymmetry to revisit).
- DBLP and ACM loaders (only IMDB implemented so far).
- Degree-/feature-aware node mass `μ` instead of uniform.
