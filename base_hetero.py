"""
Base (no coarsening) node-classification accuracy on the ORIGINAL heterogeneous
graph, for all three backbones. Used to confirm dataset identity by matching
AH-UGC Table 4's 'Base' column (e.g. DBLP author ~92-94, ACM paper ~92).
"""
import argparse
from copy import deepcopy

import torch
import torch.nn.functional as F

from data_loading_hetero import load_hetero
from models_hetero import MODELS
from eval_hetero import _orig_adj_t_dict, _offsets

BACKBONES = ["HeteroSGC", "HeteroGCN", "HeteroGCN2"]


def base_accuracy(ds, model_name, hidden=64, num_layers=3, epochs=1000,
                  lr=0.01, wd=5e-4, dtype=torch.float32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, type_order, target = ds["data"], ds["type_order"], ds["target_type"]
    relations = ds["relations"]
    fo = _offsets(ds["node_counts"])

    x = {t: ds["X_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t = _orig_adj_t_dict(relations, ds["A_global"], fo, type_order, device, dtype)
    y = data[target].y.to(device)
    valid = y >= 0
    num_classes = int(y[valid].max().item()) + 1

    train_mask = data[target].train_mask.to(device) if hasattr(data[target], "train_mask") else valid
    val_mask = data[target].val_mask.to(device) if hasattr(data[target], "val_mask") else None
    test_mask = data[target].test_mask.to(device) if hasattr(data[target], "test_mask") else valid
    train_mask = train_mask & valid
    test_mask = test_mask & valid
    if val_mask is not None:
        val_mask = val_mask & valid

    torch.manual_seed(0)
    model = MODELS[model_name]((type_order, relations), hidden, num_classes, target,
                               num_layers=num_layers).to(device)
    model.eval()
    with torch.no_grad():
        model(x, adj_t)                       # lazy Linear init
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # HGCond's train_model_ealystop: run all epochs, track the checkpoint with
    # the best validation accuracy, evaluate THAT checkpoint on test (not the
    # final-epoch weights).
    best_val_acc, best_weights = -1.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, adj_t)
        loss = F.nll_loss(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                pred = model(x, adj_t).argmax(dim=-1)
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        pred = model(x, adj_t).argmax(dim=1)
    return (pred[test_mask] == y[test_mask]).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="DBLP")
    args = p.parse_args()
    ds = load_hetero(args.dataset)
    print(f"[base] {args.dataset}: types={ds['type_order']} counts={ds['node_counts']} "
          f"target={ds['target_type']} relations={ds['relations']}", flush=True)
    for mname in BACKBONES:
        acc = base_accuracy(ds, mname)
        print(f"[base] {args.dataset} {mname}: {acc*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
