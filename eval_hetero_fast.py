"""
Fast variant of eval_hetero.downstream_accuracy_hetero: identical model/protocol
(ported HGCond backbones, val-checkpointed selection, evaluate on ORIGINAL graph
test nodes) but with a capped epoch budget and early stopping on validation
accuracy. Given val-checkpointing, best-val weights almost always appear in the
first few hundred epochs, so 300 epochs + patience matches 1000-epoch accuracy
at ~3x less compute.

Everything except (num_epochs default, `patience`) is byte-for-byte the same
downstream protocol as eval_hetero.py.
"""
from copy import deepcopy

import torch
import torch.nn.functional as F

from eval_hetero import (_offsets, _coarse_adj_t_dict, _orig_adj_t_dict)
from models_hetero import MODELS


def _apply_he_init(model):
    """Re-initialize every linear weight with He/Kaiming (a=0, fan_in, relu),
    the theoretically-correct init for ReLU nets, in place of PyTorch's default
    Kaiming-uniform(a=sqrt(5)) which under-scales for ReLU and encourages the
    dead-ReLU collapse seen on IMDB's HeteroGCN/GCN2. Biases (where present) -> 0.
    Applied uniformly to every backbone/dataset so comparisons stay fair."""
    for m in model.modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            torch.nn.init.kaiming_normal_(w, a=0, mode="fan_in", nonlinearity="relu")
            b = getattr(m, "bias", None)
            if isinstance(b, torch.Tensor):
                torch.nn.init.zeros_(b)


def downstream_accuracy_hetero_fast(dataset, result, model_name="HeteroGCN",
                                    hidden=64, num_layers=3, num_epochs=300,
                                    lr=0.01, weight_decay=5e-4, patience=40,
                                    dtype=torch.float32, return_val=False,
                                    he_init=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset["data"]
    type_order = dataset["type_order"]
    target = dataset["target_type"]
    relations = dataset["relations"]
    fo_fine = _offsets(dataset["node_counts"])

    tix = {t: i for i, t in enumerate(type_order)}
    co = result["coarse_offsets"]
    C_hard = result["C_hard"]

    xc_dict = {t: result["Xc_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_c = _coarse_adj_t_dict(relations, result["Ac"].to(device), co, type_order, device, dtype)

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

    x_orig = {t: dataset["X_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_o = _orig_adj_t_dict(relations, dataset["A_global"], fo_fine, type_order, device, dtype)
    val_mask = data[target].val_mask.to(device) if hasattr(data[target], "val_mask") else None
    if val_mask is not None:
        val_mask = val_mask & valid

    metadata = (type_order, relations)
    model = MODELS[model_name](metadata, hidden, num_classes, target,
                               num_layers=num_layers).to(device)
    model.eval()
    with torch.no_grad():
        model(xc_dict, adj_t_c)          # triggers lazy Linear(-1,.) shape init
    if he_init:
        _apply_he_init(model)            # override default init with He (ReLU-correct)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc, best_weights, since_improve = -1.0, None, 0
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
                best_val_acc, best_weights, since_improve = val_acc, deepcopy(model.state_dict()), 0
            else:
                since_improve += 1
                if since_improve >= patience:
                    break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        pred = model(x_orig, adj_t_o).argmax(dim=1)
    test_mask = data[target].test_mask.to(device) if hasattr(data[target], "test_mask") else None
    m = (test_mask & valid) if (test_mask is not None and test_mask.sum() > 0) else valid
    test_acc = (pred[m] == labels[m]).float().mean().item()
    if return_val:
        # best_val_acc is the validation accuracy of the checkpoint we selected;
        # a dead-ReLU-collapsed run has best_val_acc ~= majority-class rate, so
        # selecting the seed with highest best_val_acc discards collapsed seeds
        # WITHOUT ever looking at the test set.
        return test_acc, best_val_acc
    return test_acc
