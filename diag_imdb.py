"""
Diagnostic for the IMDB HeteroGCN / HeteroGCN2 collapse (identical 36.33%).
Runs the winning IMDB coarsening config, then trains GCN and GCN2 with
instrumentation: loss trajectory, NaN checks, coarse-label distribution,
and the PREDICTION distribution on the test set (to confirm majority-class
collapse). Also runs a quick lr ablation. Diagnostic only -- prints facts.
"""
import torch
import torch.nn.functional as F
from copy import deepcopy

from data_loading_hetero import load_hetero
from befgc_hetero import run_befgc_hetero
from models_hetero import MODELS
from eval_hetero import (_offsets, _coarse_adj_t_dict, _orig_adj_t_dict)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST = dict(alpha=0.5436249914654229, beta=8.23241538863602,
            gamma=0.02802597061804132, eta=0.020138813679897907)


def build(ds, result, dtype=torch.float32):
    type_order, target = ds["type_order"], ds["target_type"]
    relations = ds["relations"]
    tix = {t: i for i, t in enumerate(type_order)}
    co, C_hard = result["coarse_offsets"], result["C_hard"]
    fo = result["fine_offsets"]
    fo_fine = _offsets(ds["node_counts"])

    xc = {t: result["Xc_dict"][t].to(device=DEV, dtype=dtype) for t in type_order}
    adj_c = _coarse_adj_t_dict(relations, result["Ac"].to(DEV), co, type_order, DEV, dtype)

    labels = ds["data"][target].y.to(DEV)
    valid = labels >= 0
    num_classes = int(labels[valid].max().item()) + 1
    ti = tix[target]
    C_tgt = C_hard[fo[ti]:fo[ti+1], co[ti]:co[ti+1]].to(device=DEV, dtype=torch.float32)
    onehot = F.one_hot(labels.clamp(min=0), num_classes).to(torch.float32)
    onehot[~valid] = 0
    class_counts = C_tgt.t() @ onehot
    labels_c = class_counts.argmax(dim=1)
    nonempty = class_counts.sum(dim=1) > 0

    x_orig = {t: ds["X_dict"][t].to(device=DEV, dtype=dtype) for t in type_order}
    adj_o = _orig_adj_t_dict(relations, ds["A_global"], fo_fine, type_order, DEV, dtype)
    val_mask = ds["data"][target].val_mask.to(DEV) & valid
    test_mask = ds["data"][target].test_mask.to(DEV) & valid
    return (xc, adj_c, labels, labels_c, nonempty, num_classes,
            x_orig, adj_o, val_mask, test_mask, (type_order, relations), target)


def train_diag(model_name, pack, lr=0.01, wd=5e-4, epochs=1000):
    (xc, adj_c, labels, labels_c, nonempty, num_classes,
     x_orig, adj_o, val_mask, test_mask, metadata, target) = pack
    torch.manual_seed(0)
    model = MODELS[model_name](metadata, 64, num_classes, target, num_layers=3).to(DEV)
    model.eval()
    with torch.no_grad():
        model(xc, adj_c)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_w, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(xc, adj_c)
        loss = F.nll_loss(out[nonempty], labels_c[nonempty])
        loss.backward(); opt.step()
        if ep in (0, 100, 500, 999):
            print(f"    [{model_name} lr={lr}] ep{ep} loss={loss.item():.4f} "
                  f"nan={torch.isnan(loss).item()}", flush=True)
        model.eval()
        with torch.no_grad():
            pv = model(x_orig, adj_o).argmax(dim=-1)
            va = (pv[val_mask] == labels[val_mask]).float().mean().item()
        if va > best_val:
            best_val, best_w, best_ep = va, deepcopy(model.state_dict()), ep
    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        pred = model(x_orig, adj_o).argmax(dim=-1)
    acc = (pred[test_mask] == labels[test_mask]).float().mean().item()
    dist = torch.bincount(pred[test_mask], minlength=num_classes).tolist()
    print(f"  >> {model_name} lr={lr}: acc={acc*100:.2f}%  best_val={best_val*100:.2f}% "
          f"@ep{best_ep}  pred_test_dist={dist}", flush=True)
    return acc


def main():
    ds = load_hetero("IMDB")
    tgt = ds["target_type"]
    labels = ds["data"][tgt].y
    valid = labels >= 0
    nc = int(labels[valid].max().item()) + 1
    print("[diag] IMDB test-label dist:",
          torch.bincount(labels[ds['data'][tgt].test_mask & valid], minlength=nc).tolist(), flush=True)

    coarse_counts = [max(1, int(round(0.3 * n))) for n in ds["node_counts"]]
    result = run_befgc_hetero(ds["A_global"], ds["X_dict"], ds["type_order"],
                              ds["node_counts"], coarse_counts,
                              alpha=BEST["alpha"], beta=BEST["beta"], gamma=BEST["gamma"],
                              eta=BEST["eta"], T=200, seed=0, device=str(DEV),
                              dtype=torch.float64)
    pack = build(ds, result)
    # coarse label distribution
    print("[diag] coarse-label dist (nonempty):",
          torch.bincount(pack[3][pack[4]], minlength=pack[5]).tolist(),
          " n_coarse_target_nonempty=", int(pack[4].sum().item()), flush=True)

    print("[diag] === default lr=0.01 ===", flush=True)
    for m in ("HeteroGCN", "HeteroGCN2", "HeteroSGC"):
        train_diag(m, pack, lr=0.01)

    print("[diag] === ablation lr=0.003 ===", flush=True)
    for m in ("HeteroGCN", "HeteroGCN2"):
        train_diag(m, pack, lr=0.003)


if __name__ == "__main__":
    main()
