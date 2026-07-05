"""
Balanced Entropic Fused Graph Coarsening (BEFGC).

Implements Algorithm 1 from GW_coarsening.pdf: entropic mirror-descent
optimization of a fused (feature + structure) transport-based coarsening
matrix C, with an explicit KL balance penalty against supernode collapse.

Notation follows the paper: n nodes, m supernodes, mu = node mass (uniform),
C (n x m) row-stochastic coarsening matrix, P = diag(mu) C the transport
plan, nu = C^T mu the learned supernode mass.
"""
import torch


def init_C(n, m, keep_frac=0.15, floor=1e-3, seed=None, device="cpu", dtype=torch.float64):
    """Random positive, row-stochastic init satisfying C @ 1_m = 1_n.

    A sparsity mask (~keep_frac nonzero per row) mimics the prior FGC init,
    but every row keeps a small floor everywhere so no row is ever all-zero.
    """
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    base = torch.rand(n, m, generator=g).to(device=device, dtype=dtype)
    mask = (torch.rand(n, m, generator=g) <= keep_frac).to(device=device, dtype=dtype)
    C = base * mask + floor
    C = C / C.sum(dim=1, keepdim=True)
    return C


def _transport_stats(C, mu, A, X):
    """Compute P, nu, Xc, Ac for the current C.

    nu is clamped away from 0 before dividing: an empty supernode (no mass)
    would otherwise produce 0/0 = NaN in Xc/Ac that then poisons every
    downstream metric, even though that supernode is simply unused.
    """
    P = mu.unsqueeze(1) * C                      # (n, m)
    nu = P.sum(dim=0)                             # (m,)
    nu_safe = nu.clamp_min(1e-12)
    Xc = (P.t() @ X) / nu_safe.unsqueeze(1)        # (m, d)
    AP = A @ P                                     # (n, m)
    Ac = (P.t() @ AP) / (nu_safe.unsqueeze(1) * nu_safe.unsqueeze(0))  # (m, m)
    return P, nu, Xc, Ac, AP


def _feature_cost(X, Xc):
    """G_X[i, a] = ||X_i - Xc_a||^2, via the standard expansion."""
    x2 = (X * X).sum(dim=1, keepdim=True)          # (n, 1)
    xc2 = (Xc * Xc).sum(dim=1).unsqueeze(0)         # (1, m)
    cross = X @ Xc.t()                              # (n, m)
    return x2 + xc2 - 2.0 * cross


def run_befgc(
    A,
    X,
    m,
    alpha=0.5,
    beta=1.0,
    gamma=0.01,
    eta=0.05,
    T=100,
    mu=None,
    C0=None,
    seed=0,
    tol=1e-7,
    device="cpu",
    dtype=torch.float64,
    verbose=False,
):
    """Run Algorithm 1 (Balanced Entropic Fused Graph Coarsening).

    Args:
        A: (n, n) dense adjacency (or Laplacian) tensor.
        X: (n, d) dense feature tensor.
        m: target number of supernodes.
        alpha: [0,1] structure/feature tradeoff (0 = features only, 1 = structure only).
        beta: balance (KL-to-uniform) weight, >= 0.
        gamma: entropy weight, >= 0.
        eta: mirror-descent step size (constant).
        T: number of iterations.
        mu: (n,) node mass distribution; defaults to uniform 1/n.
        C0: optional (n, m) initial coarsening matrix; random init otherwise.
        tol: early-stop threshold on relative objective change.

    Returns:
        dict with C_hard (n, m) one-hot, Ac (m, m), Xc (m, d), C_soft (final
        soft C before hardening), history (list of per-iter objective values).
    """
    A = A.to(device=device, dtype=dtype)
    X = X.to(device=device, dtype=dtype)
    n = A.shape[0]
    d = X.shape[1]

    if mu is None:
        mu = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    else:
        mu = mu.to(device=device, dtype=dtype)

    C = C0.to(device=device, dtype=dtype) if C0 is not None else init_C(
        n, m, seed=seed, device=device, dtype=dtype
    )

    nu_bar = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    A2mu = (A * A) @ mu  # (n,) constant row term for the structural gradient
    term1_const = (mu * A2mu).sum()  # mu^T (A∘A) mu, constant part of structural loss

    history = []
    prev_J = None

    for t in range(T):
        P, nu, Xc, Ac, AP = _transport_stats(C, mu, A, X)

        G_X = _feature_cost(X, Xc)

        Ac2 = Ac * Ac
        # Ac.t(): the cross term in the structural gradient is A P Ac^T
        # (Peyre/Cuturi/Solomon GW tensor trick); collapses to A P Ac only
        # when Ac is symmetric, which it is for undirected A but won't be
        # once directed/heterogeneous relations are introduced.
        cross_term = AP @ Ac.t()                                # (n, m)
        col_term = Ac2 @ nu                                      # (m,)
        G_A = 2.0 * (A2mu.unsqueeze(1) - 2.0 * cross_term + col_term.unsqueeze(0))

        nu_safe = nu.clamp_min(1e-12)
        G_B = (torch.log(nu_safe / nu_bar) + 1.0).unsqueeze(0).expand(n, m)

        P_safe = P.clamp_min(1e-300)
        G_H = torch.log(P_safe)

        G = (1.0 - alpha) * G_X + alpha * G_A + beta * G_B + gamma * G_H

        # Objective value (for logging / early stopping)
        feat_loss = (P * G_X).sum()
        struct_loss = term1_const - 2.0 * (P * cross_term).sum() + (nu * col_term).sum()
        bal_loss = (nu * torch.log(nu_safe / nu_bar)).sum()
        ent_loss = (P * (torch.log(P_safe) - 1.0)).sum()
        J = (1.0 - alpha) * feat_loss + alpha * struct_loss + beta * bal_loss + gamma * ent_loss
        history.append(J.item())

        # Log-domain stabilized multiplicative (mirror) update
        logC = torch.log(C.clamp_min(1e-300))
        logits = logC - eta * G
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        C = torch.exp(logits)

        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"[BEFGC] iter {t:4d}  J={J.item():.6f}")

        if prev_J is not None and abs(prev_J - J.item()) < tol * max(1.0, abs(prev_J)):
            if verbose:
                print(f"[BEFGC] converged at iter {t}, dJ < {tol}")
            break
        prev_J = J.item()

    # Hard assignment
    hard_idx = C.argmax(dim=1)
    C_hard = torch.zeros_like(C)
    C_hard[torch.arange(n, device=device), hard_idx] = 1.0

    P_hard, nu_hard, Xc_hard, Ac_hard, _ = _transport_stats(C_hard, mu, A, X)

    return {
        "C_hard": C_hard,
        "C_soft": C,
        "Ac": Ac_hard,
        "Xc": Xc_hard,
        "nu": nu_hard,
        "history": history,
    }
