"""
Balanced Entropic Fused Graph Coarsening for HETEROGENEOUS graphs (BEFGC-hetero).

Type-isolated coarsening in the style of AH-UGC (Kataria et al. 2025), but
OPTIMIZATION-based rather than hashing-based: we learn one block-diagonal
coarsening matrix C over all node types and optimize it with the same entropic
mirror-descent objective as homogeneous BEFGC.

Key facts that make this clean:
  * C is block-diagonal by type: a node of type t may only be assigned to a
    supernode of type t. This is the "type mask" == AH-UGC's type isolation.
  * The multiplicative mirror update C_ia <- C_ia exp(-eta G_ia) preserves the
    block structure automatically (0 * exp(.) = 0), so the mask never has to be
    re-applied after initialization.
  * STRUCTURAL TERM = OPTION A (cross-type bipartite GW): the GW discrepancy is
    computed on the FULL block adjacency A_global, whose off-diagonal blocks are
    exactly the bipartite cross-type relations. With block-diagonal C, the
    coarsened Ac = C^T A C automatically has the correct per-edge-type blocks, so
    the cross-type coupling is captured directly through the coupling P.
  * FEATURE + BALANCE terms are computed PER TYPE, because feature dims differ
    across types and balance must hold within each type (not globally).

Notation mirrors the homogeneous befgc.py: N total nodes, M total supernodes,
mu node mass (uniform), P = diag(mu) C, nu = C^T mu supernode mass.
"""
import numpy as np
import scipy.sparse as sp
import torch


def _scipy_to_torch_sparse(A, device, dtype):
    A = A.tocoo()
    idx = torch.tensor(np.vstack((A.row, A.col)), dtype=torch.long, device=device)
    val = torch.tensor(A.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


def _blocks(counts):
    """Cumulative offsets, e.g. [4,5,3] -> [0,4,9,12]."""
    off = [0]
    for c in counts:
        off.append(off[-1] + c)
    return off


def init_C_hetero(node_counts, coarse_counts, keep_frac=0.15, floor=1e-3,
                  seed=None, device="cpu", dtype=torch.float64):
    """Block-diagonal, row-stochastic init. Each type-t block (n_t x m_t) is a
    random positive row-stochastic matrix; all cross-type entries are 0."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    N, M = sum(node_counts), sum(coarse_counts)
    C = torch.zeros(N, M, device=device, dtype=dtype)
    fo, co = _blocks(node_counts), _blocks(coarse_counts)
    for t in range(len(node_counts)):
        n_t, m_t = node_counts[t], coarse_counts[t]
        base = torch.rand(n_t, m_t, generator=g).to(device=device, dtype=dtype)
        mask = (torch.rand(n_t, m_t, generator=g) <= keep_frac).to(device=device, dtype=dtype)
        blk = base * mask + floor
        blk = blk / blk.sum(dim=1, keepdim=True)
        C[fo[t]:fo[t + 1], co[t]:co[t + 1]] = blk
    return C


def _balanced_hard_assign(C, fo, co, node_counts, coarse_counts, device):
    """Hard assignment that leaves NO empty supernode (per type).

    Start from per-row argmax, then for each empty supernode pull in the single
    node (from a supernode that currently has >1 member) with the highest soft
    affinity for it. Guarantees every supernode gets >=1 node whenever
    n_t >= m_t, at minimal disruption to the argmax solution.
    """
    N, M = C.shape
    C_hard = torch.zeros_like(C)
    for t in range(len(node_counts)):
        blk = C[fo[t]:fo[t + 1], co[t]:co[t + 1]]        # (n_t, m_t) soft
        n_t, m_t = blk.shape
        idx = blk.argmax(dim=1)                           # each node -> local supernode
        counts = torch.bincount(idx, minlength=m_t)
        empties = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
        for e in empties:
            movable = counts[idx] > 1                      # nodes whose supernode has spares
            if not movable.any():
                break
            aff = blk[:, e].clone()
            aff[~movable] = -float("inf")
            node = int(aff.argmax())
            counts[idx[node]] -= 1
            idx[node] = e
            counts[e] += 1
        C_hard[fo[t] + torch.arange(n_t, device=device), co[t] + idx] = 1.0
    return C_hard


def run_befgc_hetero(
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
    """Run type-isolated BEFGC. Returns the block-diagonal C plus coarsened
    artifacts. coarse_counts[t] = target #supernodes for type_order[t]."""
    N = sum(node_counts)
    M = sum(coarse_counts)
    fo, co = _blocks(node_counts), _blocks(coarse_counts)

    # Sparse block adjacency on device. For a 0/1 adjacency, A∘A = A, but we
    # square the stored values to stay correct for weighted graphs too.
    A = _scipy_to_torch_sparse(A_global, device, dtype)
    A2 = torch.sparse_coo_tensor(A.indices(), A.values() ** 2, A.shape).coalesce()

    # Per-type dense feature blocks.
    X_blocks = [X_dict[t].to(device=device, dtype=dtype) for t in type_order]

    mu = torch.full((N,), 1.0 / N, device=device, dtype=dtype)
    A2mu = torch.sparse.mm(A2, mu.unsqueeze(1)).squeeze(1)          # (N,)
    term1_const = (mu * A2mu).sum()

    # Per-type balance target: type t holds mass n_t/N spread over m_t supernodes.
    nu_bar = torch.zeros(M, device=device, dtype=dtype)
    for t in range(len(type_order)):
        n_t, m_t = node_counts[t], coarse_counts[t]
        nu_bar[co[t]:co[t + 1]] = (n_t / N) / m_t

    C = init_C_hetero(node_counts, coarse_counts, seed=seed, device=device, dtype=dtype)

    history, prev_J = [], None

    for it in range(T):
        P = mu.unsqueeze(1) * C                                     # (N, M)
        nu = P.sum(dim=0)                                            # (M,)
        nu_safe = nu.clamp_min(1e-12)

        AP = torch.sparse.mm(A, P)                                   # (N, M)
        # NOTE: keep dtype=float64. On heterogeneous graphs with a tiny node
        # type (ACM subject=73, DBLP conference=20) beside huge types, some
        # supernodes have very small mass, so Ac entries can be large and Ac^2
        # in the structural term overflows float32 -> inf -> NaN. float64 has
        # the range to handle it and RAM is not a constraint here.
        Ac = (P.t() @ AP) / (nu_safe.unsqueeze(1) * nu_safe.unsqueeze(0))   # (M, M)

        # ---- Feature term: per-type block, off-diagonal blocks left 0 (masked) ----
        G_X = torch.zeros(N, M, device=device, dtype=dtype)
        feat_loss = torch.zeros((), device=device, dtype=dtype)
        for t in range(len(type_order)):
            Xt = X_blocks[t]                                         # (n_t, d_t)
            Pt = P[fo[t]:fo[t + 1], co[t]:co[t + 1]]                 # (n_t, m_t)
            nut = nu_safe[co[t]:co[t + 1]]                           # (m_t,)
            Xct = (Pt.t() @ Xt) / nut.unsqueeze(1)                  # (m_t, d_t)
            x2 = (Xt * Xt).sum(1, keepdim=True)
            xc2 = (Xct * Xct).sum(1).unsqueeze(0)
            gxt = x2 + xc2 - 2.0 * (Xt @ Xct.t())                   # (n_t, m_t)
            G_X[fo[t]:fo[t + 1], co[t]:co[t + 1]] = gxt
            feat_loss = feat_loss + (Pt * gxt).sum()

        # ---- Structural term: cross-type GW on the full block adjacency ----
        Ac2 = Ac * Ac
        cross_term = AP @ Ac.t()                                    # (N, M); .t() matters (Ac not sym in general)
        col_term = Ac2 @ nu                                         # (M,)
        G_A = 2.0 * (A2mu.unsqueeze(1) - 2.0 * cross_term + col_term.unsqueeze(0))
        struct_loss = term1_const - 2.0 * (P * cross_term).sum() + (nu * col_term).sum()

        # ---- Balance (per-type via nu_bar) and entropy ----
        G_B = (torch.log(nu_safe / nu_bar) + 1.0).unsqueeze(0).expand(N, M)
        bal_loss = (nu * torch.log(nu_safe / nu_bar)).sum()

        P_safe = P.clamp_min(1e-300)
        G_H = torch.log(P_safe)
        ent_loss = (P * (torch.log(P_safe) - 1.0)).sum()

        G = (1.0 - alpha) * G_X + alpha * G_A + beta * G_B + gamma * G_H
        J = (1.0 - alpha) * feat_loss + alpha * struct_loss + beta * bal_loss + gamma * ent_loss
        history.append(J.item())

        # ---- Mirror (multiplicative) update; block structure preserved for free ----
        logC = torch.log(C.clamp_min(1e-300))
        logits = logC - eta * G
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        C = torch.exp(logits)

        if verbose and (it % 20 == 0 or it == T - 1):
            print(f"[BEFGC-h] iter {it:4d}  J={J.item():.6f}")

        if prev_J is not None and abs(prev_J - J.item()) < tol * max(1.0, abs(prev_J)):
            if verbose:
                print(f"[BEFGC-h] converged at iter {it}")
            break
        prev_J = J.item()

    # ---- Hard assignment ----
    if balanced_hard:
        C_hard = _balanced_hard_assign(C, fo, co, node_counts, coarse_counts, device)
    else:
        hard_idx = C.argmax(dim=1)
        C_hard = torch.zeros_like(C)
        C_hard[torch.arange(N, device=device), hard_idx] = 1.0

    # Coarsened artifacts from the hard assignment.
    P_h = mu.unsqueeze(1) * C_hard
    nu_h = P_h.sum(dim=0).clamp_min(1e-12)
    Ac_h = (P_h.t() @ torch.sparse.mm(A, P_h)) / (nu_h.unsqueeze(1) * nu_h.unsqueeze(0))
    Xc_dict = {}
    for t in range(len(type_order)):
        Pt = P_h[fo[t]:fo[t + 1], co[t]:co[t + 1]]
        nut = nu_h[co[t]:co[t + 1]]
        xc = (Pt.t() @ X_blocks[t]) / nut.unsqueeze(1)          # mass-weighted centroid
        # L2 row-normalize the coarse features to MATCH the original features
        # (which data_loading L2-normalizes to unit norm). A centroid of unit
        # vectors has norm < 1, so without this the GNN trains on small-norm
        # coarse features but is tested on unit-norm original features -- a
        # train/test scale mismatch that pushes the no-bias mean-aggregation
        # backbones (HeteroGCN/GCN2) into dead-ReLU / majority-class collapse.
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
