"""
ABLATION 4.3.1 -- Independent GW.

Question: does the JOINT cross-type coupling in BEFGC-hetero's optimization
matter, or would coarsening each node type completely independently (as its
own homogeneous BEFGC problem, blind to every other type) give the same
result?

This does NOT just zero the cross-type edges inside the existing joint loop
(that is ablation 4.3.2, "GW without cross-type relations" -- a much smaller
change). Here we remove the coupling at the OPTIMIZATION level: each type t
gets its own mu_t = 1/n_t normalization, its own nu_bar_t, its own convergence
trajectory, run to completion with zero information about any other type.
Only after every type has independently converged do we stitch the per-type
hard assignments back into one global block-diagonal C/Ac/Xc_dict, purely for
compatibility with the shared eval code.

Implementation note: rather than reimplementing the sparse GW loop, each
type's independent problem is dispatched to the EXISTING, already-verified
`run_befgc_hetero` (see befgc_hetero.py) as a single-type "heterogeneous"
problem over just that type's own diagonal adjacency block A_tt. This is
mathematically identical to a from-scratch per-type BEFGC loop (with only one
type, there are no cross-type blocks to speak of) and means this ablation
carries over befgc_hetero.py's existing gradient/invariant verification
for free, instead of needing to be re-derived and re-checked from scratch.

Does NOT modify befgc_hetero.py or any other original file.
"""
import numpy as np
import torch

from befgc_hetero import run_befgc_hetero, _blocks


def run_befgc_independent_gw(
    A_global,
    X_dict,
    type_order,
    node_counts,
    coarse_counts,
    alpha=0.5,
    beta=1.0,
    gamma=0.01,
    eta=0.3,
    T=200,
    seed=0,
    tol=1e-7,
    device="cpu",
    dtype=torch.float64,
    verbose=False,
    balanced_hard=True,
):
    """Same signature/return shape as run_befgc_hetero, so this is a drop-in
    replacement anywhere the joint version is called (eval_hetero_fast, etc).
    """
    N, M = sum(node_counts), sum(coarse_counts)
    fo, co = _blocks(node_counts), _blocks(coarse_counts)

    C_soft = torch.zeros(N, M, device=device, dtype=dtype)
    C_hard = torch.zeros(N, M, device=device, dtype=dtype)
    Ac = torch.zeros(M, M, device=device, dtype=dtype)
    Xc_dict = {}
    nu = torch.zeros(M, device=device, dtype=dtype)
    per_type_history = {}
    total_empty = 0

    for t, name in enumerate(type_order):
        A_tt = A_global[fo[t]:fo[t + 1], fo[t]:fo[t + 1]].tocsr()
        res_t = run_befgc_hetero(
            A_tt, {name: X_dict[name]}, [name],
            [node_counts[t]], [coarse_counts[t]],
            alpha=alpha, beta=beta, gamma=gamma, eta=eta, T=T,
            seed=seed, tol=tol, device=device, dtype=dtype,
            verbose=verbose, balanced_hard=balanced_hard,
        )
        n_lo, n_hi = fo[t], fo[t + 1]
        m_lo, m_hi = co[t], co[t + 1]
        C_soft[n_lo:n_hi, m_lo:m_hi] = res_t["C_soft"]
        C_hard[n_lo:n_hi, m_lo:m_hi] = res_t["C_hard"]
        Ac[m_lo:m_hi, m_lo:m_hi] = res_t["Ac"]           # off-diagonal (cross-type) stays 0
        nu[m_lo:m_hi] = res_t["nu"]
        Xc_dict[name] = res_t["Xc_dict"][name]
        per_type_history[name] = res_t["history"]
        total_empty += res_t["empty_supernodes"]

    return {
        "C_soft": C_soft,
        "C_hard": C_hard,
        "Ac": Ac,
        "Xc_dict": Xc_dict,
        "nu": nu,
        "coarse_offsets": co,
        "fine_offsets": fo,
        "type_order": type_order,
        "coarse_counts": coarse_counts,
        "empty_supernodes": total_empty,
        "history": per_type_history,   # dict per type, NOT a flat list like the joint version
    }
