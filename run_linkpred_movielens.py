"""
Full link-prediction run for MovieLens (a link-prediction-only dataset, no
node-classification labels -- see data_loading_new.py). Identical to
run_linkpred.py except it loads via `load_movielens()` instead of
`load_hetero(name)`, since MovieLens isn't in data_loading_hetero.py's
LOADERS registry (kept separate deliberately -- see data_loading_new.py's
docstring). Does not modify run_linkpred.py or any other existing file.

Usage:
    python run_linkpred_movielens.py --ratio 0.3 --he_init
"""
import argparse
import json
import time
import numpy as np
import torch

from data_loading_new import load_movielens
from befgc_hetero import run_befgc_hetero
from link_pred_hetero import sample_link_split_and_mask, train_and_eval_linkpred
from sweep_hetero import sample_params

MODELS = ["HeteroSGC", "HeteroGCN", "HeteroGCN2"]


def sweep(ds, split, type_order, node_counts, coarse_counts, max_evals, iters,
          device, num_epochs, patience, obj_seeds, coarse_dtype, he_init=False):
    rng = np.random.default_rng(0)
    best_score, best_params = -np.inf, None
    for t in range(max_evals):
        params = sample_params(rng)
        per_seed_means, diverged_any = [], False
        for sd in range(obj_seeds):
            result = run_befgc_hetero(
                split["A_masked"], ds["X_dict"], type_order, node_counts, coarse_counts,
                alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
                eta=params["eta"], T=iters, seed=sd, device=device, dtype=coarse_dtype,
            )
            diverged_any |= result["history"][-1] > result["history"][0] + 1e-6
            aucs = []
            for m in MODELS:
                torch.manual_seed(sd)
                auc, _ = train_and_eval_linkpred(ds, result, split, model_name=m,
                                                 num_epochs=num_epochs, patience=patience,
                                                 he_init=he_init)
                aucs.append(auc)
            per_seed_means.append(float(np.mean(aucs)))
        score = float(np.mean(per_seed_means))
        worst = float(np.min(per_seed_means))
        tag = ""
        if score > best_score and not diverged_any:
            best_score, best_params = score, params
            tag = "  <-- best"
        print(f"  [linkpred-ml sweep {t+1:02d}/{max_evals}] mean_auc={score*100:5.2f}% "
              f"worstseed={worst*100:5.2f}% empty={result['empty_supernodes']} "
              f"(a={params['alpha']:.2f} b={params['beta']:.2f} "
              f"g={params['gamma']:.4f} e={params['eta']:.3f})"
              f"{' DIVERGED' if diverged_any else ''}{tag}", flush=True)
    return best_params, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--max_evals", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--obj_seeds", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--coarse_dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--he_init", action="store_true")
    p.add_argument("--out_jsonl", default="linkpred_gw_MovieLens.jsonl")
    args = p.parse_args()

    coarse_dtype = torch.float64 if args.coarse_dtype == "float64" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[linkpred-ml] device={device}, ratio={args.ratio}, "
          f"val_frac={args.val_frac}, test_frac={args.test_frac}, sweep={args.max_evals}, "
          f"obj_seeds={args.obj_seeds}, epochs={args.num_epochs}, patience={args.patience}, "
          f"coarse_dtype={args.coarse_dtype}", flush=True)

    t0 = time.time()
    ds = load_movielens()
    type_order, node_counts = ds["type_order"], ds["node_counts"]
    coarse_counts = [max(1, int(round(args.ratio * n))) for n in node_counts]

    split = sample_link_split_and_mask(ds, val_frac=args.val_frac,
                                       test_frac=args.test_frac, seed=0)
    print(f"\n[linkpred-ml] === MovieLens === rel={split['rel']} "
          f"train/val/test={len(split['train_pos'])}/{len(split['val_pos'])}/"
          f"{len(split['test_pos'])} node_counts={node_counts} "
          f"coarse={coarse_counts}", flush=True)

    best_params, best_auc = sweep(ds, split, type_order, node_counts, coarse_counts,
                                  args.max_evals, args.iters, device,
                                  args.num_epochs, args.patience, args.obj_seeds,
                                  coarse_dtype, he_init=args.he_init)
    print(f"[linkpred-ml] MovieLens best robust AUC={best_auc*100:.2f}% "
          f"params={best_params}", flush=True)

    per_model_auc = {m: [] for m in MODELS}
    per_model_ap = {m: [] for m in MODELS}
    empties = []
    for sd in range(args.n_seeds):
        result = run_befgc_hetero(
            split["A_masked"], ds["X_dict"], type_order, node_counts, coarse_counts,
            alpha=best_params["alpha"], beta=best_params["beta"],
            gamma=best_params["gamma"], eta=best_params["eta"],
            T=args.iters, seed=sd, device=device, dtype=coarse_dtype,
        )
        empties.append(result["empty_supernodes"])
        row = []
        for m in MODELS:
            torch.manual_seed(sd)
            auc, ap = train_and_eval_linkpred(ds, result, split, model_name=m,
                                              num_epochs=args.num_epochs,
                                              patience=args.patience, he_init=args.he_init)
            per_model_auc[m].append(auc)
            per_model_ap[m].append(ap)
            row.append(f"{m.replace('Hetero','')}: AUC={auc*100:.1f} AP={ap*100:.1f}")
        print(f"[linkpred-ml]   seed {sd}: " + " | ".join(row) +
              f" empty={result['empty_supernodes']}", flush=True)

    rec = {"dataset": "MovieLens", "ratio": args.ratio, "task": "link_prediction",
           "rel": list(split["rel"]), "best_params": best_params,
           "avg_empty": float(np.mean(empties)), "n_seeds": args.n_seeds,
           "mins": (time.time() - t0) / 60, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[linkpred-ml] === MovieLens RESULT (avg_empty={np.mean(empties):.1f}) ===", flush=True)
    for m in MODELS:
        a = np.array(per_model_auc[m]) * 100
        p_ = np.array(per_model_ap[m]) * 100
        rec[m + "_auc_mean"], rec[m + "_auc_std"] = float(a.mean()), float(a.std())
        rec[m + "_ap_mean"], rec[m + "_ap_std"] = float(p_.mean()), float(p_.std())
        print(f"[linkpred-ml]   {m:11s}: AUC={a.mean():.2f}+/-{a.std():.2f}  "
              f"AP={p_.mean():.2f}+/-{p_.std():.2f}", flush=True)
    with open(args.out_jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")

    print("\n[linkpred-ml] ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
