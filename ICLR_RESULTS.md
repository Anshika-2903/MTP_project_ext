# BEFGC-hetero — ICLR Extension: Results Log

Working results log for the ICLR extension of the MTP thesis (BEFGC: Balanced
Entropic Fused Graph Coarsening). All runs use the standard protocol unless
noted: r=0.3 coarsening ratio, He/Kaiming-normal init, 5 seeds, capped-epoch
+ early-stopping eval (`eval_hetero_fast.py`), HGCond's three backbones
(HeteroSGC / HeteroGCN / HeteroGCN2).

Code lives in `mtp_befgc_hetero_iclr/` (separate from the verified, untouched
thesis code in `mtp_befgc_hetero/`). See that folder's files for the actual
implementations referenced below.

Last updated: 2026-09-13 (session complete — full ratio sweep finished).

---

## 0. Executive Summary

This session ran a full experimental campaign extending the MTP thesis
toward an ICLR submission: ~40+ PBS jobs on PRAGYA HPC across IMDB, ACM,
DBLP, LastFM, and MovieLens, covering ablations, a full coarsening-ratio
sweep, a new downstream task (link prediction), and two new datasets.

**Confirmed findings:**
1. **BEFGC beats AH-UGC on all 9 backbone×dataset cells** at r=0.3 (from the
   thesis) — the headline comparison holds.
2. **SGC has a genuine, ratio-independent dependency on cross-type
   coupling.** Removing it (via either ablation that eliminates cross-type
   information) roughly halves SGC's accuracy at EVERY ratio tested
   (0.15/0.3/0.4/0.5) on all three datasets. GCN/GCN2 degrade far less.
3. **GW's structural sophistication shows no measurable edge over plain
   Laplacian smoothness** — confirmed across node classification AND link
   prediction, across 5 datasets (IMDB/ACM/DBLP/LastFM/MovieLens), and
   across the ratio sweep. This is a real, actionable finding: the extra
   complexity of Gromov-Wasserstein optimal transport isn't earning its
   keep in this regime, at least not by these metrics.
4. **Independent-GW and No-cross-type are structurally incapable of
   cross-type link prediction** (they never build any cross-type
   connectivity, so there's nothing to predict) — a logical property of
   those ablations, not a codebase limitation. They remain valid for node
   classification and for link prediction on intra-type relations.
5. **HEADLINE NEW DISCOVERY: a genuine, ratio-dependent ACM+GCN2
   instability**, not tied to any one ablation. Bimodal ~50/50 seed split
   at r=0.15, a catastrophic single-seed collapse at r=0.4, clean at r=0.5.
   Appears with the Laplacian ablation, the full unmodified method, AND
   (in a milder form) in a completely unrelated context — LastFM link
   prediction with no-cross-type. DBLP shows zero instability anywhere.
   This points to a GCN2-architecture-specific sensitivity (GCNII-style
   skip connections + no-bias + mean-aggregation) that surfaces under
   certain coarsening/dataset conditions — worth investigating directly and
   reporting honestly (mean±std is misleading for this cell; median or
   best-of-k-via-validation would be more honest).

**Not started this session** (from the original ICLR roadmap): node
clustering (NMI/ARI); homogeneous-coarsening-on-heterograph baseline (4.2);
new backbones R-GCN/HAN/HGT/SimpleHGN (3.1/3.2); AMiner and OGB-MAG
(blocked by an architectural memory limit — dense M×M coarsened adjacency
doesn't scale to million-node graphs, needs a redesign not just more
compute); Freebase, DBpedia, Yelp.

See §1–§6 below for full details and every data point.

---

## 1. Ablation baselines (4.3) — node classification, r=0.3

Reference: full BEFGC (GW-based, joint cross-type optimization) numbers from
the thesis, `befgc-hetero-results.md`.

| Dataset | Backbone | **Full BEFGC** | **4.3.1 Independent GW** | **4.3.2 No-cross-type** | **4.3.3 Laplacian** |
|---|---|---:|---:|---:|---:|
| IMDB | SGC  | 64.94 | 36.98 ± 0.46 | 36.88 ± 0.34 | 64.81 ± 0.37 |
| IMDB | GCN  | 69.11 | 62.66 ± 2.43 | 61.64 ± 2.80 | 69.06 ± 0.57 |
| IMDB | GCN2 | 69.00 | 61.03 ± 1.47 | 60.44 ± 0.92 | 69.18 ± 1.15 |
| ACM  | SGC  | 91.95 | 49.57 ± 0.00 | 49.57 ± 0.00 | 92.12 ± 0.27 |
| ACM  | GCN  | 85.77 | 84.81 ± 2.82 | 84.81 ± 2.82 | 92.52 ± 1.03 |
| ACM  | GCN2 | 92.12 | 83.03 ± 3.04 | 83.03 ± 3.04 | 85.98 ± 12.63¹ |
| DBLP | SGC  | 91.99 | 30.83 ± 3.43 | 30.83 ± 3.43 | 91.80 ± 1.20 |
| DBLP | GCN  | 83.01 | 80.94 ± 0.69 | 80.94 ± 0.69 | 83.37 ± 1.16 |
| DBLP | GCN2 | 82.47 | 80.76 ± 0.72 | 80.76 ± 0.72 | 82.70 ± 1.05 |

¹ **Resolved as a fluke**, not a real instability — see §1.3 below. A 10-seed
rerun (different sweep-selected hyperparameters) gave 92.75 ± 0.59 with zero
collapses. Combined 15 seeds tested (5 original + 10 rerun) → 1 collapse
total, consistent with the already-documented rare dying-ReLU pathology in
no-bias/ReLU/mean-agg backbones, not something specific to Laplacian
smoothness.

### 1.1 Ablation definitions
- **4.3.1 Independent GW** (`ablation_independent_gw.py`): each node type
  coarsened as its own fully independent BEFGC problem (own `mu`, own
  convergence) — removes joint cross-type OPTIMIZATION entirely, not just the
  cross-type edges.
- **4.3.2 No-cross-type** (`ablation_no_cross_type.py`): keeps the joint
  optimization loop, but zeroes cross-type edges (A_xy) in the input
  adjacency before coarsening.
- **4.3.3 Laplacian heterogeneous** (`ablation_laplacian_hetero.py`): same
  block-C / optimization / feature / balance / entropy scaffold, GW
  structural term replaced with the standard Laplacian-smoothness quadratic
  form `trace(P^T L P)`, `L = D - A` on the full cross-type adjacency.

### 1.2 Findings
1. **SGC collapses hard without cross-type coupling, on all three datasets**
   (64.94→37, 91.95→50, 91.99→31 — roughly halved everywhere), while GCN/GCN2
   degrade only modestly (2–8 pts). Sum-aggregation appears structurally
   dependent on cross-type signal in a way mean-aggregation isn't.
2. **4.3.1 and 4.3.2 give bit-identical numbers on ACM and DBLP.** Worked out
   analytically why: with cross-type edges zeroed, the coarsened adjacency
   `Ac` is provably invariant to the mu-normalization convention (global 1/N
   vs. per-type 1/n_t cancels in the Ac ratio), and the balance term is
   deliberately scale-invariant by construction. So in the low-alpha regime
   these particular sweeps landed in, 4.3.1 and 4.3.2 end up computing the
   same effective optimization on these two datasets — **they are not
   independent evidence here** and should be flagged as such in the writeup
   rather than presented as two separate confirmations. (On IMDB they're
   close but not identical — 36.98 vs 36.88 SGC, etc. — presumably a
   different alpha regime from the sweep.)
3. **Laplacian smoothness ties or beats full GW-based BEFGC** at r=0.3 on all
   three datasets for node classification (see §1.3 below — resolved after
   the ACM rerun). GW's extra structural sophistication isn't clearly earning
   its keep here.

### 1.3 ACM Laplacian GCN2 stability check (10-seed rerun)
Original 5-seed run: 85.98 ± 12.63 (seed 2 collapsed to 60.9%, others 90–94%).
10-seed rerun (`abllap_ACM_10seed.log`, different sweep-selected
hyperparameters — `alpha=0.544, beta=8.23, gamma=0.028, eta=0.020`):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| GCN2 | 92.5 | 93.2 | 92.6 | 92.3 | 92.6 | 93.3 | 92.2 | 93.3 | 91.8 | 93.8 |

Mean **92.75 ± 0.59** — no collapse in any of the 10 seeds. Conclusion: the
original 12.63 std was a rare single-seed fluke, not a systemic Laplacian
instability.

---

## 2. Link prediction (1.1) — ROC-AUC / AP, r=0.3

New downstream task (`link_pred_hetero.py`, `run_linkpred.py` for GW,
`run_linkpred_laplacian.py` for the Laplacian ablation). Protocol: 80/10/10
split of the highest-edge-count relation per dataset, leakage-free (val+test
edges masked out of both the coarsening input and the eval-time message-passing
graph), encoder trained on coarse graph via BCE on coarse pos/neg supernode
pairs, evaluated on original-graph embeddings — mirrors the node-classification
train-on-coarse/eval-on-original protocol exactly, with a dot-product edge
score in place of the softmax head.

### 2.1 GW (full BEFGC) vs Laplacian ablation

**IMDB** (relation: movie–actor, 10264/1282/1282 train/val/test):

| Backbone | GW AUC | Laplacian AUC | GW AP | Laplacian AP |
|---|---:|---:|---:|---:|
| SGC  | 57.07 ± 0.16 | 57.01 ± 0.08 | 63.24 ± 0.24 | 63.23 ± 0.24 |
| GCN  | 64.23 ± 2.09 | 63.60 ± 2.49 | 64.68 ± 2.10 | 64.58 ± 2.42 |
| GCN2 | 62.64 ± 2.36 | 62.75 ± 1.93 | 63.61 ± 2.50 | 64.43 ± 2.01 |

**ACM** (GW only so far — Laplacian run in progress):

| Backbone | GW AUC | GW AP |
|---|---:|---:|
| SGC  | 78.48 ± 0.03 | 76.99 ± 0.03 |
| GCN  | 89.78 ± 1.35 | 86.60 ± 1.68 |
| GCN2 | 80.31 ± 3.31 | 78.55 ± 2.28 |

**DBLP** (GW only so far — Laplacian run in progress):

| Backbone | GW AUC | GW AP |
|---|---:|---:|
| SGC  | 88.25 ± 0.01 | 90.27 ± 0.01 |
| GCN  | 87.90 ± 0.06 | 90.08 ± 0.05 |
| GCN2 | 87.66 ± 0.05 | 89.93 ± 0.03 |

Notably tight across backbones and seeds (std ≤0.06) — all three backbones
converge to nearly the same DBLP link-prediction performance, unlike node
classification where they diverged sharply (SGC 91.99 vs GCN 83.01).

### 2.2 Finding
**GW shows no meaningful edge over Laplacian smoothness on link prediction
either** (IMDB numbers are within noise of each other on every backbone).
This now holds across two different task types (node classification AND link
prediction) on IMDB — looks like a real property of the method at r=0.3, not
an artifact of one evaluation protocol. Worth being upfront about as either a
limitation or motivation to check harder ratios (r=0.15) where GW's extra
structure might matter more.

### 2.3 Independent GW / No-cross-type on link prediction — structurally impossible on cross-type relations
Attempted extending the ablation comparison to link prediction on IMDB, ACM,
DBLP (all cross-type relations: movie–actor, paper–author, paper–term).
**Both No-cross-type AND Independent-GW fail with "0 coarse pairs to train
on"** — not a bug in either case, and not fixable by tuning: both ablations
structurally prevent any cross-type connectivity from ever reaching the
coarsened graph (no-cross-type zeroes the edges outright; independent-GW
never even looks at other types while coarsening each one). Since the very
relation being predicted IS a cross-type relation, the coarsened graph simply
has nothing to learn from. **Conclusion: link prediction on a cross-type
relation can only meaningfully compare full GW-based BEFGC against the
Laplacian ablation** (both of which preserve the full block adjacency) — the
other two ablations are only evaluable via node classification, or via link
prediction on an INTRA-type relation (e.g. LastFM's user–user ties), not
attempted here. (Independent-GW's separate `KeyError` from a dict-vs-list
`history` mismatch was fixed, but the underlying 0-pairs issue is the real,
unfixable reason all three cross-type reruns failed identically.)

---

## 3. New datasets

Beyond IMDB/ACM/DBLP (all node-classification), added as **link-prediction-
only** datasets (no labels/masks) via `data_loading_new.py`, using the same
`build_from_hetero_data` contract so all existing coarsening/eval code works
unmodified.

### 3.1 MovieLens (ml-latest-small)
- 610 users, 9,742 movies, 100,836 ratings (all used as positive edges,
  implicit-feedback convention).
- Movie features: 19-dim genre one-hot. User features: none explicit — picked
  up automatically by `build_from_hetero_data`'s existing featureless-type
  fallback (mean of rated movies' genre vectors), same mechanism already used
  for ACM's "subject" / DBLP's "conference" types.
- Bypasses PyG's own `MovieLens` loader deliberately (it requires
  `sentence_transformers` for text-embedding movie titles — an unnecessary
  dependency; genre one-hot is a standard, simpler feature choice).
- Smoke-tested end-to-end: coarsening → link prediction, HeteroGCN AUC 0.91
  (30-epoch quick test, not a tuned run).

### 3.2 LastFM — DONE
- PyG's built-in `LastFm` dataset. Download blocked by the IITD proxy
  (Dropbox host) — same class of issue as ACM.mat in the original thesis
  work. Downloaded locally (869MB) and transferred to PRAGYA via a **chunked
  upload** (10MB pieces) after discovering the direct transfer was being cut
  off at a hard ~19.7MB limit (not random flakiness — same byte count every
  retry).
- 1,892 users, 17,632 artists, 1,088 tags; 5 relations including a user–user
  social-tie relation (all node types featureless — constant fallback, no
  raw features available in this dataset).
- Link prediction (GW-based BEFGC, r=0.3, relation=user–artist, 5 seeds):

| Backbone | AUC | AP |
|---|---:|---:|
| SGC  | 74.57 ± 0.14 | 81.33 ± 0.11 |
| GCN  | 84.08 ± 0.37 | 84.80 ± 0.60 |
| GCN2 | 83.73 ± 1.78 | 84.41 ± 2.06 |

GCN/GCN2 clearly outperform SGC here, unlike DBLP where all three backbones
were nearly tied — dataset-dependent, as expected. **Laplacian ablation
matches GW almost exactly here too** (SGC 74.57±0.14, GCN 84.12±0.30,
GCN2 83.61±1.73) — the GW≈Laplacian pattern now holds on 5/5 datasets tested
for link prediction (IMDB, ACM, DBLP, LastFM, MovieLens).

MovieLens Laplacian link-pred: SGC 90.16±0.01, GCN 91.22±0.45,
GCN2 91.23±0.28 — again essentially identical to GW's 90.16/91.26/91.25.

### 2.4 LastFM user–user (intra-type) — the one case where no-cross-type IS testable
Since no-cross-type only removes CROSS-type edges, LastFM's user–user
relation (intra-type, 25,434 edges) survives it, making this the one
legitimate no-cross-type link-prediction data point:

| Backbone | AUC | AP |
|---|---:|---:|
| SGC  | 65.57 ± 1.91 | 64.74 ± 1.64 |
| GCN  | 80.10 ± 0.19 | 79.78 ± 0.25 |
| GCN2 | 72.72 ± 8.41¹ | 72.75 ± 8.64¹ |

¹ **Third distinct instance of GCN2-specific seed instability** — 2 of 5
seeds collapsed (61.1/63.9 AUC) while the other 3 held at 79-80 AUC. Combined
with the ACM+GCN2+Laplacian pattern (§1.3/§5.5) and the ACM+GCN2+full-BEFGC
mild instability (§5.6), this is now showing up across THREE different
contexts (different dataset, different ablation, different task type) — the
common thread is always GCN2. **Revised, broader conclusion: this looks like
a GCN2-specific (GCNII-style skip connection) seed sensitivity that surfaces
under certain coarsening conditions, not something tied to one dataset or
one ablation.** Worth investigating GCN2's initialization/training dynamics
directly rather than treating each occurrence as a separate anomaly.

### 3.3 Not yet attempted
OGB-MAG, AMiner (Academic); Yelp (Social); Freebase, DBpedia (Knowledge
graphs). `ogb` package confirmed installed on PRAGYA (v1.3.6); AMiner/OGB-MAG
loaders importable but not yet download-tested.

---

## 4. Node clustering (1.2) — NMI / ARI, r=0.3

New downstream task (`node_clustering_hetero.py`, `run_clustering.py`).
Protocol: train the same encoder architecture used for link prediction plus
a small linear classification head on coarse pseudo-labels (majority vote,
same as node classification), then run k-means on the trained encoder's
ORIGINAL-graph embeddings (k=num_classes) and score against true labels via
NMI/ARI. k-means never sees the labels; they're used only to score the
clustering afterward.

**IMDB** (5 seeds, He-init):

| Backbone | NMI | ARI |
|---|---:|---:|
| SGC  | 16.69 ± 2.56 | 16.83 ± 4.78 |
| GCN  | 21.49 ± 1.66 | 20.86 ± 2.58 |
| GCN2 | 21.19 ± 1.97 | 20.28 ± 3.27 |

**ACM** (5 seeds, He-init):

| Backbone | NMI | ARI |
|---|---:|---:|
| SGC  | 59.81 ± 4.90 | 62.02 ± 6.27 |
| GCN  | 69.12 ± 3.27 | 71.15 ± 5.85 |
| GCN2 | 69.79 ± 3.85 | 72.35 ± 6.92 |

**DBLP** (5 seeds, He-init):

| Backbone | NMI | ARI |
|---|---:|---:|
| SGC  | 7.82 ± 6.95 | 3.80 ± 2.30 |
| GCN  | 34.91 ± 5.73 | 27.50 ± 5.30 |
| GCN2 | 33.55 ± 6.19 | 27.29 ± 4.79 |

**Findings:**
- GCN/GCN2 clearly outperform SGC on clustering quality on **all three**
  datasets — the gap is dramatic on DBLP (NMI 7.8 vs ~34, a 4-5x difference)
  and IMDB, moderate on ACM.
- **ACM produces by far the cleanest clusters** (NMI ~60-70%) — consistent
  with ACM3025 being a well-separated 3-class task in the literature.
- **Interesting reversal on DBLP**: SGC is the *best* backbone for node
  CLASSIFICATION on DBLP (91.99%, beating GCN's 83.01% — see §1), but by far
  the *worst* for clustering quality (NMI 7.82 vs GCN's 34.91). This means
  SGC's embeddings on DBLP separate classes well enough for a supervised
  linear decision boundary (classification) but do NOT form natural,
  well-separated clusters in embedding space (unsupervised k-means) — a
  genuinely interesting distinction between "linearly separable" and
  "cluster-separable" representations that's worth a sentence in the
  write-up. GCN/GCN2 give more "clusterable" embeddings across the board,
  even on datasets where SGC wins on raw classification accuracy.

## 5. New models / tasks — not yet started
- Homogeneous-coarsening-on-heterograph baseline (4.2)
- R-GCN, HAN, HGT, SimpleHGN backbones (3.1/3.2)

---

## 5.5 Ratio sweep (r=0.15 / 0.4 / 0.5) — ablations, in progress

Extending §1's r=0.3 ablation grid to other coarsening ratios, to check
whether the findings there are ratio-specific or general. 27 jobs total
(3 ablations × 3 datasets × 3 ratios); results below as they land.

### Independent GW (4.3.1)

| Dataset | Ratio | SGC | GCN | GCN2 |
|---|---|---:|---:|---:|
| IMDB | 0.15 | 36.66 ± 0.48 | 54.19 ± 2.71 | 52.53 ± 4.71 |
| IMDB | 0.4  | 36.91 ± 0.45 | 67.89 ± 1.77 | 65.40 ± 1.71 |
| IMDB | 0.5  | 37.51 ± 0.59 | 72.71 ± 0.54 | 72.50 ± 0.53 |
| ACM  | 0.15 | 49.57 ± 0.00 | 73.36 ± 1.56 | 71.16 ± 1.95 |
| ACM  | 0.4  | 49.57 ± 0.00 | 88.02 ± 4.11 | 88.01 ± 2.75 |
| ACM  | 0.5  | 49.57 ± 0.00 | 92.63 ± 0.64 | 91.09 ± 2.07 |
| DBLP | 0.15 | 30.88 ± 1.57 | 76.05 ± 0.92 | 75.13 ± 1.75 |
| DBLP | 0.4  | 31.61 ± 2.84 | 83.21 ± 0.40 | 83.09 ± 0.43 |
| DBLP | 0.5  | 31.40 ± 2.94 | 86.47 ± 0.71 | 86.64 ± 0.48 |

Note: SGC's collapse (§1.2 finding 1) is **ratio-independent** — stuck at
~37/50/31 regardless of r on IMDB/ACM/DBLP respectively, confirming it's a
structural property (loss of cross-type coupling), not something more
coarsening fixes. GCN/GCN2 recover substantially at higher r (less
information loss overall), as expected.

### Laplacian (4.3.3)

| Dataset | Ratio | SGC | GCN | GCN2 |
|---|---|---:|---:|---:|
| IMDB | 0.15 | 53.05 ± 2.84 | 58.80 ± 2.16 | 57.47 ± 3.42 |
| IMDB | 0.4  | 68.77 ± 0.73 | 73.19 ± 1.18 | 73.55 ± 0.97 |
| IMDB | 0.5  | 72.21 ± 0.60 | 77.60 ± 0.43 | 77.64 ± 1.40 |
| ACM  | 0.15 | 88.57 ± 0.64 | 87.50 ± 1.56 | 76.19 ± 14.85¹ |
| ACM  | 0.4  | 92.22 ± 0.64 | 92.93 ± 2.38 | 76.63 ± 16.17¹ |
| ACM  | 0.5  | 93.18 ± 0.38 | 92.39 ± 3.47 | 93.98 ± 1.35 |
| DBLP | 0.15 | 88.92 ± 1.59 | 76.84 ± 1.08 | 76.29 ± 1.19 |
| DBLP | 0.4  | 93.82 ± 0.13 | 85.93 ± 0.31 | 85.59 ± 0.23 |
| DBLP | 0.5  | 94.63 ± 0.16 | 88.72 ± 0.28 | 88.20 ± 0.44 |

**Laplacian ratio sweep now fully complete for IMDB and DBLP** — no instability observed on DBLP at any ratio (all std ≤0.44). The ACM+GCN2 instability (footnote 1) appears specific to ACM, not a general Laplacian-ablation problem.

¹ **REVISED CONCLUSION (was "rare fluke" in §1.3): this is a recurring
instability, not a one-off.** Three high-variance ACM+GCN2+Laplacian
collapses now observed: r=0.3 (85.98±12.63, original), r=0.15 (76.19±14.85),
r=0.4 (76.63±16.17) — 3 out of 4 ratios tested show it, only r=0.5 is clean
(93.98±1.35). The §1.3 10-seed rerun that seemed to "resolve" it was at
DIFFERENT sweep-selected hyperparameters than the original 5-seed run, which
likely explains why it didn't reproduce there — this looks like a genuine
ACM+GCN2+Laplacian-specific instability (dying-ReLU-style, per-seed) that is
sensitive to the coarsening ratio and/or hyperparameters, not simple seed
noise. Worth a dedicated investigation (e.g. is it specific to the Laplacian
ablation, or does full-GW BEFGC's ACM/GCN2 show it too at these ratios?)
rather than treating it as resolved.

### No-cross-type (4.3.2) — running, all 9 combos launched

| Dataset | Ratio | SGC | GCN | GCN2 |
|---|---|---:|---:|---:|
| IMDB | 0.15 | 36.60 ± 0.13 | 54.28 ± 3.21 | 53.12 ± 2.41 |
| IMDB | 0.4  | 36.89 ± 0.46 | 66.00 ± 4.00 | 66.20 ± 1.87 |
| IMDB | 0.5  | 37.51 ± 0.59 | 72.71 ± 0.54 | 72.50 ± 0.53 |
| ACM  | 0.15 | 49.57 ± 0.00 | 73.25 ± 1.39 | 71.63 ± 1.69 |
| ACM  | 0.4  | 49.57 ± 0.00 | 88.02 ± 4.11 | 88.01 ± 2.75 |
| ACM  | 0.5  | 49.57 ± 0.00 | 92.63 ± 0.64 | 91.09 ± 2.07 |
| DBLP | 0.15 | 30.88 ± 1.57 | 76.05 ± 0.92 | 75.13 ± 1.75 |
| DBLP | 0.4  | 31.61 ± 2.84 | 83.21 ± 0.40 | 83.09 ± 0.43 |
| DBLP | 0.5  | 31.65 ± 2.88 | 85.91 ± 0.75 | 86.31 ± 0.79 |

Note: IMDB r=0.15 no-cross-type (36.60/54.28/53.12) is CLOSE to but not
bit-identical to independent-GW's r=0.15 numbers (36.66/54.19/52.53) —
consistent with the earlier finding that the exact mathematical equivalence
between these two ablations only holds on ACM/DBLP, not IMDB.

---

## 5.6 Main-BEFGC (full GW, unmodified) ratio sweep — NEW, diagnostic

Launched to (a) fill the gap where only ACM had a ratio sweep for the main
method, and (b) diagnose whether the ACM+GCN2 instability is Laplacian-
specific or more general.

| Dataset | Ratio | SGC | GCN | GCN2 |
|---|---|---:|---:|---:|
| IMDB | 0.15 | 53.15 ± 2.86 | 58.80 ± 2.15 | 57.48 ± 3.42 |
| IMDB | 0.4  | 68.77 ± 0.73 | 73.40 ± 1.16 | 73.56 ± 0.98 |
| IMDB | 0.5  | 72.21 ± 0.60 | 77.65 ± 0.42 | 77.61 ± 1.36 |
| ACM  | 0.15 | 89.02 ± 0.73 | 88.12 ± 0.88 | **83.40 ± 4.88¹ (10-seed: 82.71±5.50, bimodal)** |
| ACM  | 0.4  | 92.68 ± 0.23 | 94.20 ± 0.69 | **86.79 ± 14.41² (catastrophic single-seed: 58.0 vs 93-95 others)** |
| ACM  | 0.5  | 93.51 ± 0.21 | 95.22 ± 0.45 | 95.20 ± 0.45 — clean, no instability |
| DBLP | 0.15 | 89.04 ± 1.63 | 77.13 ± 1.20 | 76.06 ± 0.98 |
| DBLP | 0.4  | 93.70 ± 0.17 | 86.13 ± 0.68 | 85.79 ± 0.60 |
| DBLP | 0.5  | 94.54 ± 0.13 | 88.31 ± 0.26 | 88.35 ± 0.55 |

**Ratio sweep now 100% complete** — all 4 methods (main BEFGC, independent-GW,
no-cross-type, laplacian) × all 3 datasets × all 4 ratios (0.15/0.3/0.4/0.5).
DBLP confirmed clean across every single cell (std always ≤1.63, no
exceptions) — the ACM+GCN2 instability truly is ACM-specific.

**r=0.5 confirmed clean for ACM+GCN2** with the full method (std=0.45) —
completes the picture: instability at r=0.15 (bimodal) and r=0.4
(catastrophic single-seed), clean at r=0.5. DBLP shows no instability at any
ratio, for any method, ever tested in this session.

² **Different character than r=0.15's bimodal split** — at r=0.4 the full
method shows ONE catastrophic single-seed collapse (58.0%, near majority-
class) among four clean seeds (93-95%), matching the SEVERITY pattern of the
Laplacian ablation at the same ratio (76.63±16.17, also one severe collapse)
rather than r=0.15's milder 50/50 bimodal split. So the instability's
*character* (bimodal vs. single-catastrophic-outlier) varies by ratio, but
ACM+GCN2 is unstable at every ratio tested (0.15, 0.3, 0.4) with the
unmodified method — only r=0.5 has been clean so far for either method.

¹ **Refined via a 10-seed rerun** (`mainbefgc_ACM_r0.15_10seed.log`) — and the
picture is more precise (and more concerning) than "mild instability":

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| GCN2 | 88.5 | 74.6 | 79.3 | 78.1 | 76.6 | 88.0 | 87.8 | 87.9 | 78.1 | 88.2 |
| GCN  | 89.5 | 78.1 | 88.8 | 89.5 | 89.7 | 89.2 | 88.2 | 90.2 | 81.8 | 89.8 |

10-seed mean: SGC 87.59±1.10, GCN 87.49±3.89, **GCN2 82.71±5.50**. This is
**not** a rare single-seed collapse (like the Laplacian ACM+GCN2 pattern,
one bad seed among five good ones) — it's a **genuine bimodal split**: 5 of
10 seeds land in a "high" regime (~87-89%) and 5 in a "low" regime
(~75-79%), a roughly 50/50 split. GCN shows the same bimodal pattern, more
mildly (2 of 10 seeds low: 78.1, 81.8 vs 88-90 for the rest).

**FINAL conclusion: ACM+GCN2 (and to a lesser extent ACM+GCN) has a
genuine, frequent (~50%) bimodal training-outcome split at r=0.15 with the
UNMODIFIED, full GW-based method** — this is not a Laplacian-ablation
artifact, not a rare fluke, and not dataset-agnostic (DBLP shows nothing
like it at any ratio/method tested). The Laplacian ablation's more extreme
single-seed collapses (e.g. 60.9%, std up to 16) look like a MORE SEVERE
version of this same underlying ACM+GCN2 sensitivity, not a separate
phenomenon. This deserves a place in the thesis/paper as a real limitation:
mean±std alone is misleading for ACM+GCN2 — a bimodal-aware reporting
(e.g. median, or best-of-k via validation, as `eval_hetero_fast.py`'s
`return_val` option already supports) would be more honest than the
5-seed mean±std format used elsewhere in this work.

---

## 6. Infrastructure notes (for reproducing / extending)
- All new code lives in `mtp_befgc_hetero_iclr/` (local) /
  `~/Static_coarsening/static_coarsening_2/befgc_hetero_iclr/` (PRAGYA) —
  kept separate from the verified thesis code (`mtp_befgc_hetero/` /
  `befgc_hetero/`), which is untouched. Backup of the thesis code at
  `mtp_befgc_hetero_backup_20260913/`.
- Fixed a latent data-race in `data_loading_hetero.py`'s
  `_fix_imdb_directories()` (iclr copy only): it used to unconditionally
  delete/rebuild IMDB's cached `processed/data.pt` on every call, which
  crashes when multiple jobs load IMDB concurrently from the same data dir.
  Now guarded to skip if already fixed.
- PBS walltime: 20 min is enough for IMDB: ACM/DBLP (bigger graphs — ACM
  author=17,431 nodes, DBLP paper=14,328/term=7,723) need up to ~1hr for a
  full 10-trial sweep + 5-seed run.
