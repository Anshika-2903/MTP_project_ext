"""
ABLATION 4.3.3 -- Laplacian heterogeneous.

Question: how much of BEFGC's benefit comes specifically from the
Gromov-Wasserstein structural term, versus just having ANY structure-aware
term at all? This keeps the exact same framework as befgc_hetero.py --
same block-diagonal C, same feature/balance/entropy terms, same mirror-descent
optimization loop, same hard-assignment step -- and swaps ONLY the structural
term: GW is replaced by the standard Laplacian-smoothness quadratic form

    J_struct = trace(P^T L P) = sum_{(i,j) in E} A_ij * ||P_i,: - P_j,:||^2

where L = D - A is the unnormalized graph Laplacian of the SAME full
(cross-type-inclusive) block adjacency A_global used by the joint method, and
P = diag(mu) C is the transport plan. This is the classic "connected nodes
should end up with similar coarse-assignment profiles" smoothness penalty used
in spectral graph coarsening (Loukas & Vandergheynst-style), as opposed to
GW's "the induced coarse graph should look like the original graph" fidelity
criterion.

Gradient: dJ_struct/dP = 2 L P = 2(D P - A P). D is a fixed per-node degree
computed once from A (constant across iterations), and A P is already needed
elsewhere in the loop, so this term is CHEAPER than the GW term (no M x M
Ac/Ac^2/cross_term/col_term construction at all).

Feature/balance/entropy terms and their gradients are copied verbatim from
befgc_hetero.py (already verified there) -- only the structural block differs.
Does NOT modify befgc_hetero.py or any other original file.
"""
import numpy as np
import scipy.sparse as sp
import torch

from befgc_hetero import _scipy_to_torch_sparse, _blocks, init_C_hetero, _balanced_hard_assign


def run_befgc_hetero_laplacian(
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
    """Same signature/return shape as run_befgc_hetero (befgc_hetero.py) --
    drop-in replacement anywhere the GW joint version is called."""
    N = sum(node_counts)
    M = sum(coarse_counts)
    fo, co = _blocks(node_counts), _blocks(coarse_counts)

    A = _scipy_to_torch_sparse(A_global, device, dtype)
    deg = torch.sparse.mm(A, torch.ones(N, 1, device=device, dtype=dtype)).squeeze(1)  # (N,)

    X_blocks = [X_dict[t].to(device=device, dtype=dtype) for t in type_order]

    mu = torch.full((N,), 1.0 / N, device=device, dtype=dtype)

    nu_bar = torch.zeros(M, device=device, dtype=dtype)
    for t in range(len(type_order)):
        n_t, m_t = node_counts[t], coarse_counts[t]
        nu_bar[co[t]:co[t + 1]] = (n_t / N) / m_t

    C = init_C_hetero(node_counts, coarse_counts, seed=seed, device=device, dtype=dtype)

    history, prev_J = [], None

    for it in range(T):
        P = mu.unsqueeze(1) * C
        nu = P.sum(dim=0)
        nu_safe = nu.clamp_min(1e-12)

        AP = torch.sparse.mm(A, P)                                  # (N, M)

        # ---- Feature term (identical to befgc_hetero.py) ----
        G_X = torch.zeros(N, M, device=device, dtype=dtype)
        feat_loss = torch.zeros((), device=device, dtype=dtype)
        for t in range(len(type_order)):
            Xt = X_blocks[t]
            Pt = P[fo[t]:fo[t + 1], co[t]:co[t + 1]]
            nut = nu_safe[co[t]:co[t + 1]]
            Xct = (Pt.t() @ Xt) / nut.unsqueeze(1)
            x2 = (Xt * Xt).sum(1, keepdim=True)
            xc2 = (Xct * Xct).sum(1).unsqueeze(0)
            gxt = x2 + xc2 - 2.0 * (Xt @ Xct.t())
            G_X[fo[t]:fo[t + 1], co[t]:co[t + 1]] = gxt
            feat_loss = feat_loss + (Pt * gxt).sum()

        # ---- Structural term: Laplacian smoothness in place of GW ----
        LP = deg.unsqueeze(1) * P - AP                              # L @ P, L = D - A
        struct_loss = (P * LP).sum()
        G_A = 2.0 * LP

        # ---- Balance and entropy (identical to befgc_hetero.py) ----
        G_B = (torch.log(nu_safe / nu_bar) + 1.0).unsqueeze(0).expand(N, M)
        bal_loss = (nu * torch.log(nu_safe / nu_bar)).sum()

        P_safe = P.clamp_min(1e-300)
        G_H = torch.log(P_safe)
        ent_loss = (P * (torch.log(P_safe) - 1.0)).sum()

        G = (1.0 - alpha) * G_X + alpha * G_A + beta * G_B + gamma * G_H
        J = (1.0 - alpha) * feat_loss + alpha * struct_loss + beta * bal_loss + gamma * ent_loss
        history.append(J.item())

        logC = torch.log(C.clamp_min(1e-300))
        logits = logC - eta * G
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        C = torch.exp(logits)

        if verbose and (it % 20 == 0 or it == T - 1):
            print(f"[BEFGC-h-lap] iter {it:4d}  J={J.item():.6f}")

        if prev_J is not None and abs(prev_J - J.item()) < tol * max(1.0, abs(prev_J)):
            if verbose:
                print(f"[BEFGC-h-lap] converged at iter {it}")
            break
        prev_J = J.item()

    if balanced_hard:
        C_hard = _balanced_hard_assign(C, fo, co, node_counts, coarse_counts, device)
    else:
        hard_idx = C.argmax(dim=1)
        C_hard = torch.zeros_like(C)
        C_hard[torch.arange(N, device=device), hard_idx] = 1.0

    # Ac reported here is the ACTUAL induced coarse adjacency (P_h^T A P_h),
    # for eval/reporting/comparability with the GW version -- it was never
    # part of THIS method's optimization objective (which used the Laplacian
    # smoothness term instead), it's just recorded as a downstream artifact.
    P_h = mu.unsqueeze(1) * C_hard
    nu_h = P_h.sum(dim=0).clamp_min(1e-12)
    Ac_h = (P_h.t() @ torch.sparse.mm(A, P_h)) / (nu_h.unsqueeze(1) * nu_h.unsqueeze(0))
    Xc_dict = {}
    for t in range(len(type_order)):
        Pt = P_h[fo[t]:fo[t + 1], co[t]:co[t + 1]]
        nut = nu_h[co[t]:co[t + 1]]
        xc = (Pt.t() @ X_blocks[t]) / nut.unsqueeze(1)
        xn = xc.norm(p=2, dim=1, keepdim=True)
        xn[xn == 0] = 1.0
        Xc_dict[type_order[t]] = xc / xn

    empty = int((C_hard.sum(dim=0) == 0).sum().item())

    return {
        "C_soft": C,
        "C_hard": C_hard,
        "Ac": Ac_h,
        "Xc_dict": Xc_dict,
        "nu": nu_h,
        "coarse_offsets": co,
        "fine_offsets": fo,
        "type_order": type_order,
        "coarse_counts": coarse_counts,
        "empty_supernodes": empty,
        "history": history,
    }
