"""
Evaluation for BEFGC-hetero, following the AH-UGC / HGCond protocol:
train a heterogeneous GNN (real HGCond backbone) on the coarsened graph, then
evaluate on the ORIGINAL graph's test nodes for the target (labelled) type.

HGCond models consume normalized sparse adjacencies (`adj_t @ h`), one per edge
type, where adj_t = asymmetric_gcn_norm(A) has shape (n_dst, n_src). We build
those for both the coarsened graph (from Ac blocks) and the original graph
(from the global block adjacency A_global).

Training config = HGCond default: lr=0.01, weight_decay=5e-4, 1000 epochs,
hidden=64, num_layers=3, no dropout.
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from models_hetero import MODELS, asymmetric_gcn_norm


def _offsets(counts):
    off = [0]
    for c in counts:
        off.append(off[-1] + c)
    return off


def _adj_t_from_dense(block_src_dst, device, dtype, thresh=1e-6):
    """block is (n_src, n_dst) dense; return normalized adj_t (n_dst, n_src)."""
    bt = block_src_dst.t().contiguous().to(dtype)
    bt = bt.clone()
    bt[bt.abs() < thresh] = 0
    adj_t = SparseTensor.from_dense(bt).to(device)
    return asymmetric_gcn_norm(adj_t)


def _adj_t_from_scipy(block_src_dst, device, dtype):
    """block is (n_src, n_dst) scipy; return normalized adj_t (n_dst, n_src)."""
    bt = block_src_dst.T.tocoo()
    row = torch.from_numpy(bt.row).long()
    col = torch.from_numpy(bt.col).long()
    val = torch.from_numpy(bt.data).to(dtype)
    adj_t = SparseTensor(row=row, col=col, value=val,
                         sparse_sizes=(bt.shape[0], bt.shape[1])).to(device)
    return asymmetric_gcn_norm(adj_t)


def _coarse_adj_t_dict(relations, Ac, co, type_order, device, dtype):
    tix = {t: i for i, t in enumerate(type_order)}
    d = {}
    for rel in relations:
        s, _, dst = rel
        i, j = tix[s], tix[dst]
        block = Ac[co[i]:co[i + 1], co[j]:co[j + 1]]        # (m_src, m_dst)
        d[rel] = _adj_t_from_dense(block, device, dtype)
    return d


def _orig_adj_t_dict(relations, A_global, fo, type_order, device, dtype):
    tix = {t: i for i, t in enumerate(type_order)}
    d = {}
    for rel in relations:
        s, _, dst = rel
        i, j = tix[s], tix[dst]
        block = A_global[fo[i]:fo[i + 1], fo[j]:fo[j + 1]]  # (n_src, n_dst) scipy
        d[rel] = _adj_t_from_scipy(block, device, dtype)
    return d


def downstream_accuracy_hetero(dataset, result, model_name="HeteroGCN",
                               hidden=64, num_layers=3, num_epochs=1000,
                               lr=0.01, weight_decay=5e-4, dtype=torch.float32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset["data"]
    type_order = dataset["type_order"]
    target = dataset["target_type"]
    relations = dataset["relations"]
    fo_fine = _offsets(dataset["node_counts"])

    tix = {t: i for i, t in enumerate(type_order)}
    co = result["coarse_offsets"]
    C_hard = result["C_hard"]

    # ---- coarse graph: features + normalized adjacencies ----
    xc_dict = {t: result["Xc_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_c = _coarse_adj_t_dict(relations, result["Ac"].to(device), co, type_order, device, dtype)

    # ---- coarse target labels via majority vote (labeled members only) ----
    labels = data[target].y.to(device)
    valid = labels >= 0
    num_classes = int(labels[valid].max().item()) + 1
    ti = tix[target]
    fo = result["fine_offsets"]
    C_tgt = C_hard[fo[ti]:fo[ti + 1], co[ti]:co[ti + 1]].to(device=device, dtype=torch.float32)
    onehot = F.one_hot(labels.clamp(min=0), num_classes).to(torch.float32)
    onehot[~valid] = 0
    class_counts = C_tgt.t() @ onehot
    labels_c = class_counts.argmax(dim=1)
    nonempty = class_counts.sum(dim=1) > 0

    # ---- original graph (needed for val-checkpoint selection + final test) ----
    x_orig = {t: dataset["X_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_o = _orig_adj_t_dict(relations, dataset["A_global"], fo_fine, type_order, device, dtype)
    val_mask = data[target].val_mask.to(device) if hasattr(data[target], "val_mask") else None
    if val_mask is not None:
        val_mask = val_mask & valid

    metadata = (type_order, relations)
    model = MODELS[model_name](metadata, hidden, num_classes, target,
                               num_layers=num_layers).to(device)
    # lazy Linear(-1, .) init: run one forward before building the optimizer
    model.eval()
    with torch.no_grad():
        model(xc_dict, adj_t_c)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # HGCond's train_model_ealystop: track the checkpoint with the best
    # validation accuracy (evaluated on the ORIGINAL graph's val nodes, since
    # that's what the trained model is ultimately judged on), not the final
    # epoch's weights.
    best_val_acc, best_weights = -1.0, None
    for _ in range(num_epochs):
        model.train()
        opt.zero_grad()
        out = model(xc_dict, adj_t_c)
        loss = F.nll_loss(out[nonempty], labels_c[nonempty])
        loss.backward()
        opt.step()

        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                pred_val = model(x_orig, adj_t_o).argmax(dim=-1)
                val_acc = (pred_val[val_mask] == labels[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # ---- evaluate on ORIGINAL graph, target test nodes ----
    model.eval()
    with torch.no_grad():
        pred = model(x_orig, adj_t_o).argmax(dim=1)
    test_mask = data[target].test_mask.to(device) if hasattr(data[target], "test_mask") else None
    m = (test_mask & valid) if (test_mask is not None and test_mask.sum() > 0) else valid
    return (pred[m] == labels[m]).float().mean().item()
