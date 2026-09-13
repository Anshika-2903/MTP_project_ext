# BEFGC-hetero

**Balanced Entropic Fused Graph Coarsening**, extended to heterogeneous
graphs. This is an MTP (dual-degree) thesis project at IIT Delhi, extended
further for an ICLR submission cycle. BEFGC is a Gromov-Wasserstein
optimal-transport graph coarsening method: it learns a soft assignment of
nodes to a smaller number of "supernodes" by jointly optimizing feature
similarity, structural (GW) similarity, a balance penalty against
supernode collapse, and an entropy regularizer, via entropic mirror
descent.

It is benchmarked against **AH-UGC** (Kataria et al. 2025, hashing-based
coarsening) using **HGCond**'s GNN backbones (HeteroSGC / HeteroGCN /
HeteroGCN2), on the heterogeneous graphs IMDB, ACM, and DBLP, plus MovieLens
and LastFM added later as link-prediction-only datasets.

**Full experimental results and findings:** see [`ICLR_RESULTS.md`](ICLR_RESULTS.md)
— every number, every ablation, every finding from the full campaign, with
explanations of *why* each result looks the way it does.

---

## Two generations of code in this repo

This repo went through two phases, and the file layout reflects that:

- **Original thesis code** (verified, unmodified, submitted for the MTP
  endterm) — the core method, models, data loaders, and the main
  node-classification benchmark against AH-UGC. Files listed under
  [Core method](#core-method-original-thesis-code) through
  [Drivers](#drivers-original-thesis-code) below.
- **ICLR extension code** (added afterward, in its own files, never
  editing the originals except one documented bug fix — see
  [What's new](#whats-new-iclr-extension)) — three ablation studies, a full
  coarsening-ratio sweep, two new downstream tasks (link prediction, node
  clustering), and two new datasets (MovieLens, LastFM).

If you only want to reproduce the thesis's headline result, you only need
the "original thesis code" files. If you want the full ICLR-cycle
ablations/tasks/datasets, you need both.

---

## Quick start — which script do I run?

| I want to... | Run this | Notes |
|---|---|---|
| Reproduce the main BEFGC vs AH-UGC benchmark | `run_hetero_fast.py --datasets IMDB --ratio 0.3 --he_init` | Original thesis driver. `--datasets` accepts `IMDB`/`ACM`/`DBLP` (or several). |
| Run the "Independent GW" ablation (4.3.1) | `run_ablation_independent_gw.py --datasets ACM --ratio 0.3 --he_init` | |
| Run the "No-cross-type" ablation (4.3.2) | `run_ablation_no_cross_type.py --datasets ACM --ratio 0.3 --he_init` | |
| Run the "Laplacian" ablation (4.3.3) | `run_ablation_laplacian.py --datasets ACM --ratio 0.3 --he_init` | |
| Link prediction (GW-based, on IMDB/ACM/DBLP) | `run_linkpred.py --datasets IMDB --ratio 0.3 --he_init` | |
| Link prediction, Laplacian ablation | `run_linkpred_laplacian.py --datasets IMDB --ratio 0.3 --he_init` | Same idea for `run_linkpred_independent_gw.py` / `run_linkpred_no_cross_type.py`, but see the caveat below. |
| Link prediction on MovieLens | `run_linkpred_movielens.py --ratio 0.3 --he_init` | MovieLens/LastFM have no `--datasets` flag — one dataset per script. |
| Link prediction on LastFM | `run_linkpred_lastfm.py --ratio 0.3 --he_init` | |
| Node clustering (NMI/ARI) | `run_clustering.py --datasets IMDB --ratio 0.3 --he_init` | Only works on IMDB/ACM/DBLP (need ground-truth labels to score against). |

All scripts share the same CLI shape: `--ratio` (coarsening ratio),
`--n_seeds` (final multi-seed run, default 5), `--max_evals` (hyperparameter
sweep trials, default 10), `--he_init` (use He/Kaiming init — recommended,
fixes a dying-ReLU issue in the no-bias backbones), `--out_jsonl` (where to
append one JSON result line per dataset).

**Caveat on cross-ablation link prediction:** `run_linkpred_independent_gw.py`
and `run_linkpred_no_cross_type.py` will fail with a "0 coarse pairs" error
on IMDB/ACM/DBLP's default relations — this is *expected*, not a bug. Both
ablations remove all cross-type structure before coarsening, and the
relations being predicted (movie–actor, paper–author, paper–term) are
cross-type, so there's nothing left to predict. They only work on an
intra-type relation, e.g. `run_linkpred_lastfm_nocross_useruser.py`
(LastFM's user–user relation). See `ICLR_RESULTS.md` §2.3 for the full
explanation.

---

## Core method (original thesis code)
- **`befgc.py`** — homogeneous BEFGC (Cora/CiteSeer-scale). Entropic
  mirror-descent on a soft coarsening matrix, hand-derived closed-form
  gradients (not autograd) for numerical stability at scale.
- **`befgc_hetero.py`** — the heterogeneous extension. Block-diagonal
  coarsening matrix (one block per node type), structural term = GW on the
  full multi-relational block adjacency (cross-type edges captured directly
  through the coupling).
- **`models_hetero.py`** — `HeteroSGC` / `HeteroGCN` / `HeteroGCN2`, ported
  verbatim from the official HGCond repo (not reimplemented), since AH-UGC
  benchmarks on these exact architectures.

## Data loading (original thesis code)
- **`data_loading.py`** — Cora/CiteSeer (homogeneous).
- **`data_loading_hetero.py`** — IMDB/DBLP/ACM. Builds the symmetric block
  adjacency from all edge types; featureless node types get features via
  mean-of-neighbor propagation. *(One bug fix here for the ICLR extension —
  see [What's new](#whats-new-iclr-extension).)*

## Evaluation (original thesis code)
- **`eval.py`** — homogeneous evaluation.
- **`eval_hetero.py`** — full-protocol heterogeneous evaluation (1000
  epochs, no early stopping) — train on the coarsened graph, test on the
  original graph.
- **`eval_hetero_fast.py`** — same protocol, capped epochs + early stopping
  + He-init option. This is what all the driver scripts actually use.

## Drivers (original thesis code)
- **`run_hetero_fast.py`** — the main entry point: samples hyperparameters
  via random search, evaluates all 3 backbones per config, re-runs the best
  config at `--n_seeds` seeds, writes results to `--out_jsonl`.
- **`sweep_hetero.py`** — the hyperparameter sampler used by every driver.
- `run_hetero_full.py`, `base_hetero.py`, `benchmark_hetero.py`,
  `run_imdb_valselect.py`, `diag_imdb.py`, `diag2_configs.py`,
  `main_hetero.py` — older/diagnostic variants from earlier phases of the
  thesis (dying-ReLU investigation, full-protocol runs, etc.) — kept for
  reference, superseded by `run_hetero_fast.py` for day-to-day use.
- `sync_to_pragya.ps1` / `pull_results.ps1` / `run_on_pragya.sh` —
  Windows↔HPC sync scripts for our specific cluster (PRAGYA, IIT Delhi).
  Cluster-specific; adapt the paths for your own environment.

  The dozens of old `.pbs` job scripts accumulated during thesis
  development (`run_acm_fast.pbs`, `run_imdb_quick.pbs`, etc. — all just
  the same 2-3 driver scripts called with different flags) have been
  removed from this repo as clutter. If you're on a PBS cluster, a job file
  is just:
  ```bash
  #!/bin/bash
  #PBS -N my_job_name
  #PBS -P <your_project_code>
  #PBS -l select=1:ncpus=4:ngpus=1
  #PBS -l walltime=01:00:00
  #PBS -j oe
  #PBS -o my_job_name.log

  source <path_to_conda>/etc/profile.d/conda.sh
  conda activate <your_env>
  cd <path_to_this_repo>
  python3 -u <driver_script.py> --datasets IMDB --ratio 0.3 --he_init --out_jsonl my_results.jsonl
  ```
  Swap in whichever driver script and flags you need from the
  [Quick start](#quick-start--which-script-do-i-run) table above. This
  pattern is identical for every script in this repo, old and new.

---

## What's new (ICLR extension)

### Ablations (§1 in `ICLR_RESULTS.md`)
- **`ablation_independent_gw.py`** — each node type coarsened as its own
  fully independent BEFGC problem, no joint cross-type optimization at all.
- **`ablation_no_cross_type.py`** — keeps the joint optimization, but
  zeroes cross-type edges in the input adjacency before coarsening.
- **`ablation_laplacian_hetero.py`** — same optimization scaffold, but the
  GW structural term is replaced with a plain Laplacian-smoothness
  quadratic form.
- **`run_ablation_independent_gw.py`** / **`run_ablation_laplacian.py`** /
  **`run_ablation_no_cross_type.py`** — drivers for each, mirroring
  `run_hetero_fast.py`'s structure exactly.

### Downstream task: link prediction (§2 in `ICLR_RESULTS.md`)
- **`link_pred_hetero.py`** — the core library: leakage-free train/val/test
  edge splitting, embedding-only encoder variants of the three backbones,
  and the train-on-coarse/eval-on-original training loop. Everything else
  link-prediction-related imports from here.
- **`run_linkpred.py`** — GW-based BEFGC, for IMDB/ACM/DBLP.
- **`run_linkpred_laplacian.py`** / **`run_linkpred_independent_gw.py`** /
  **`run_linkpred_no_cross_type.py`** — same task, swapping in each
  ablation's coarsening function.
- **`run_linkpred_movielens.py`** / **`run_linkpred_movielens_laplacian.py`**
  — MovieLens-specific (it isn't in `data_loading_hetero.py`'s dataset
  registry, so it gets its own driver rather than a `--datasets` flag).
- **`run_linkpred_lastfm.py`** / **`run_linkpred_lastfm_laplacian.py`** /
  **`run_linkpred_lastfm_nocross_useruser.py`** — LastFM-specific; the last
  one targets LastFM's user–user relation specifically (see the caveat
  above).

### Downstream task: node clustering (§ new in this session)
- **`node_clustering_hetero.py`** — trains the same encoder architecture
  used for link prediction (reused, not reimplemented) plus a small linear
  classification head on coarse pseudo-labels, then clusters the trained
  encoder's embeddings on the *original* graph with k-means and scores
  against true labels via NMI/ARI. k-means itself never sees the labels —
  they're used only to score the clustering afterward.
- **`run_clustering.py`** — driver, same structure as every other task.

### New datasets (§3 in `ICLR_RESULTS.md`)
- **`data_loading_new.py`** — `load_movielens()` and `load_lastfm()`,
  returning the same dict shape as `data_loading_hetero.py`'s loaders so
  every existing coarsening/eval/task file works with them unmodified.
  Both are link-prediction-only (no classification labels).
  - MovieLens uses a custom lightweight loader (genre one-hot features)
    instead of PyG's own `MovieLens` class, to avoid an unnecessary
    `sentence_transformers` dependency for text-embedding movie titles.
  - LastFM uses PyG's built-in loader; getting the raw data onto our
    compute cluster required a workaround for a network transfer
    limitation — see `ICLR_RESULTS.md` §3.2 if you hit the same issue.

### One documented bug fix in original code
- **`data_loading_hetero.py`**'s `_fix_imdb_directories()` used to
  unconditionally delete and rebuild IMDB's cached `processed/data.pt` on
  *every* call. That's fine for one job at a time, but crashes when
  multiple jobs load IMDB concurrently from the same data directory (a race
  that only surfaces once you start running many experiments in parallel,
  which is exactly what the ICLR extension does). Fixed by guarding it to
  skip once the raw files are already in place.

---

## Results

All experimental results — every table, every finding, the full ratio
sweep, and a write-up of a genuine ACM+GCN2 training instability discovered
during this work — are in **[`ICLR_RESULTS.md`](ICLR_RESULTS.md)**.

A polished PDF/HTML summary suitable for sharing outside this repo is in
`scratch/` (`BEFGC_Results_Summary.pdf`, `results_artifact.html`) if
present — regenerate with `scratch/make_pdf.py`.

---

## Known limitations
- **Scale ceiling on the structural term.** `befgc_hetero.py`'s GW
  structural loss materializes a dense M×M coarsened-adjacency matrix every
  iteration (M = total supernodes). This is fine at the scale of
  IMDB/ACM/DBLP (M in the thousands), but makes million-node graphs like
  OGB-MAG or the full AMiner dump infeasible without an architectural
  change (sparse or low-rank approximation of the structural term) — not
  something more compute time alone fixes.
- **ACM+GCN2 training instability.** A genuine, ratio-dependent instability
  (not a bug) was found in the HeteroGCN2 backbone specifically on ACM —
  see `ICLR_RESULTS.md`'s "Key New Finding" section for the full
  characterization and a suggested reporting fix (median/best-of-k instead
  of naive mean±std for this specific cell).

## Environment
Conda env `fgc_comp`: PyTorch, PyTorch Geometric 2.2.0, `torch_sparse`,
`ogb` 1.3.6, scikit-learn 1.1.2. Developed and run on PRAGYA HPC (IIT
Delhi) via PBS batch jobs — see the `*.pbs` files for job templates and
`sync_to_pragya.ps1`/`pull_results.ps1` for the Windows↔cluster workflow we
used; adapt paths/scheduler directives for your own setup.
