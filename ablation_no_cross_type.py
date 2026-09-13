"""
ABLATION 4.3.2 -- GW without cross-type relations.

Question: do the actual cross-type edges (A_xy) matter, given the SAME joint
optimization as the full method (shared mu over all N nodes, shared nu_bar
setup, one single mirror-descent loop over the whole block-diagonal C)? This
is deliberately a much smaller change than 4.3.1 (Independent GW), which also
removes the shared joint optimization itself. Here only the input adjacency is
modified -- every other part of run_befgc_hetero (befgc_hetero.py) is used
completely unmodified.

Because Ac = P^T A P with block-diagonal P, zeroing A's off-diagonal (cross-
type) blocks automatically makes the coarsened Ac block-diagonal too, so no
change to the loss/gradient code is needed at all -- only the adjacency that
gets passed in.

Does NOT modify befgc_hetero.py or any other original file.
"""
import scipy.sparse as sp

from befgc_hetero import run_befgc_hetero, _blocks


def _drop_cross_type_edges(A_global, node_counts):
    """Zero every off-block (cross-type) entry, keep intra-type (A_xx) blocks."""
    fo = _blocks(node_counts)
    A = A_global.tolil()
    T = len(node_counts)
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            A[fo[i]:fo[i + 1], fo[j]:fo[j + 1]] = 0.0
    return A.tocsr()


def run_befgc_no_cross_type(
    A_global,
    X_dict,
    type_order,
    node_counts,
    coarse_counts,
    **kwargs,
):
    """Same signature/return shape as run_befgc_hetero -- drop-in replacement.
    All kwargs (alpha, beta, gamma, eta, T, seed, tol, device, dtype, verbose,
    balanced_hard) are forwarded unchanged to the untouched joint optimizer."""
    A_no_cross = _drop_cross_type_edges(A_global, node_counts)
    return run_befgc_hetero(
        A_no_cross, X_dict, type_order, node_counts, coarse_counts, **kwargs
    )
