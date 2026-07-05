"""
Decisive diagnostic: does ANY coarsening config avoid HeteroGCN/GCN2 collapse
on IMDB? Sweeps several (alpha,beta,gamma,eta) configs -- including low-balance
/ low-entropy ones the SGC-only sweep never picks -- and for each reports all 3
backbones' accuracy + prediction distribution (collapse = all one class).

If some config makes all 3 work  -> collapse is tuning-fixable (multi-backbone
objective). If NONE do -> deeper eval/model issue.
Short epochs (collapse is visible by ep ~100).
"""
import argparse
import torch
import torch.nn.functional as F
from copy import deepcopy

from data_loading_hetero import load_hetero
from befgc_hetero import run_befgc_hetero
from models_hetero import MODELS
from eval_hetero import _offsets, _coarse_adj_t_dict, _orig_adj_t_dict

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = [
    ("SGC-winner ", dict(alpha=0.5436, beta=8.232, gamma=0.0280, eta=0.0201)),
    ("low-bal/ent", dict(alpha=0.50,  beta=0.50,  gamma=0.0010, eta=0.05)),
    ("feat-heavy ", dict(alpha=0.20,  beta=1.00,  gamma=0.0050, eta=0.05)),
    ("struct-hevy", dict(alpha=0.80,  beta=1.00,  gamma=0.0050, eta=0.05)),
    ("vlow-ent   ", dict(alpha=0.50,  beta=1.00,  gamma=0.0005, eta=0.03)),
]


def build(ds, result, dtype=torch.float32):
    type_order, target = ds["type_order"], ds["target_type"]
    relations = ds["relations"]
    tix = {t: i for i, t in enumerate(type_order)}
    co, C_hard, fo = result["coarse_offsets"], result["C_hard"], result["fine_offsets"]
    fo_fine = _offsets(ds["node_counts"])

    xc = {t: result["Xc_dict"][t].to(device=DEV, dtype=dtype) for t in type_order}
    adj_c = _coarse_adj_t_dict(relations, result["Ac"].to(DEV), co, type_order, DEV, dtype)

    labels = ds["data"][target].y.to(DEV)
    valid = labels >= 0
    nc = int(labels[valid].max().item()) + 1
    ti = tix[target]
    C_tgt = C_hard[fo[ti]:fo[ti+1], co[ti]:co[ti+1]].to(device=DEV, dtype=torch.float32)
    onehot = F.one_hot(labels.clamp(min=0), nc).to(torch.float32); onehot[~valid] = 0
    cc = C_tgt.t() @ onehot
    labels_c = cc.argmax(dim=1); nonempty = cc.sum(dim=1) > 0

    x_orig = {t: ds["X_dict"][t].to(device=DEV, dtype=dtype) for t in type_order}
    adj_o = _orig_adj_t_dict(relations, ds["A_global"], fo_fine, type_order, DEV, dtype)
    vm = ds["data"][target].val_mask.to(DEV) & valid
    tm = ds["data"][target].test_mask.to(DEV) & valid
    return (xc, adj_c, labels, labels_c, nonempty, nc, x_orig, adj_o, vm, tm,
            (type_order, relations), target)


def train(model_name, pack, lr=0.01, wd=5e-4, epochs=500):
    (xc, adj_c, labels, labels_c, nonempty, nc, x_orig, adj_o, vm, tm, meta, tgt) = pack
    torch.manual_seed(0)
    model = MODELS[model_name](meta, 64, nc, tgt, num_layers=3).to(DEV)
    model.eval()
    with torch.no_grad():
        model(xc, adj_c)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_w = -1.0, None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.nll_loss(model(xc, adj_c)[nonempty], labels_c[nonempty])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = (model(x_orig, adj_o).argmax(1)[vm] == labels[vm]).float().mean().item()
        if va > best_val:
            best_val, best_w = va, deepcopy(model.state_dict())
    model.load_state_dict(best_w); model.eval()
    with torch.no_grad():
        pred = model(x_orig, adj_o).argmax(1)
    acc = (pred[tm] == labels[tm]).float().mean().item()
    dist = torch.bincount(pred[tm], minlength=nc).tolist()
    collapsed = (max(dist) == sum(dist))
    return acc, collapsed, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="IMDB")
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    ds = load_hetero(args.dataset)
    cc = [max(1, int(round(args.ratio * n))) for n in ds["node_counts"]]
    print(f"[diag2] {args.dataset} counts={ds['node_counts']} coarse={cc}", flush=True)

    for name, cfg in CONFIGS:
        result = run_befgc_hetero(ds["A_global"], ds["X_dict"], ds["type_order"],
                                  ds["node_counts"], cc, T=200, seed=0,
                                  device=str(DEV), dtype=torch.float64, **cfg)
        pack = build(ds, result)
        row = []
        for m in ("HeteroSGC", "HeteroGCN", "HeteroGCN2"):
            acc, col, dist = train(m, pack, epochs=args.epochs)
            row.append(f"{m.replace('Hetero',''):5s}={acc*100:5.2f}%{'[COLLAPSE]' if col else '         '}")
        print(f"[{name}] empty={result['empty_supernodes']}  " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
