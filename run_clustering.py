"""
Runner for downstream task 1.2 (Node clustering: NMI, ARI), using full
GW-based BEFGC coarsening. Same sweep-then-multiseed structure as
run_hetero_fast.py (node classification) and run_linkpred.py (link
prediction) -- the sweep objective here is mean NMI over `obj_seeds`
coarsening seeds and all three backbones.

Only works on datasets that HAVE ground-truth node-classification labels
(IMDB/ACM/DBLP) -- labels are used solely to SCORE the clustering (NMI/ARI),
never to train the k-means step itself. Does not apply to the link-
prediction-only datasets (MovieLens/LastFM), which have no labels.

Usage:
    python run_clustering.py --datasets IMDB --ratio 0.3 --he_init
"""
import argparse
import json
import time
import numpy as np
import torch

from data_loading_hetero import load_hetero
from befgc_hetero import run_befgc_hetero
from node_clustering_hetero import train_and_eval_clustering
from sweep_hetero import sample_params

MODELS = ["HeteroSGC", "HeteroGCN", "HeteroGCN2"]


def sweep(ds, type_order, node_counts, coarse_counts, max_evals, iters,
          device, num_epochs, patience, obj_seeds, coarse_dtype, he_init=False):
    rng = np.random.default_rng(0)
    best_score, best_params = -np.inf, None
    for t in range(max_evals):
        params = sample_params(rng)
        per_seed_means, diverged_any = [], False
        for sd in range(obj_seeds):
            result = run_befgc_hetero(
                ds["A_global"], ds["X_dict"], type_order, node_counts, coarse_counts,
                alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
                eta=params["eta"], T=iters, seed=sd, device=device, dtype=coarse_dtype,
            )
            diverged_any |= result["history"][-1] > result["history"][0] + 1e-6
            nmis = []
            for m in MODELS:
                torch.manual_seed(sd)
                nmi, _ = train_and_eval_clustering(ds, result, model_name=m,
                                                   num_epochs=num_epochs, patience=patience,
                                                   he_init=he_init)
                nmis.append(nmi)
            per_seed_means.append(float(np.mean(nmis)))
        score = float(np.mean(per_seed_means))
        worst = float(np.min(per_seed_means))
        tag = ""
        if score > best_score and not diverged_any:
            best_score, best_params = score, params
            tag = "  <-- best"
        print(f"  [clustering sweep {t+1:02d}/{max_evals}] mean_nmi={score*100:5.2f}% "
              f"worstseed={worst*100:5.2f}% empty={result['empty_supernodes']} "
              f"(a={params['alpha']:.2f} b={params['beta']:.2f} "
              f"g={params['gamma']:.4f} e={params['eta']:.3f})"
              f"{' DIVERGED' if diverged_any else ''}{tag}", flush=True)
    return best_params, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ACM"])
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--max_evals", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--obj_seeds", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--coarse_dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--he_init", action="store_true")
    p.add_argument("--out_jsonl", default="clustering.jsonl")
    args = p.parse_args()

    coarse_dtype = torch.float64 if args.coarse_dtype == "float64" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clustering] device={device}, datasets={args.datasets}, ratio={args.ratio}, "
          f"sweep={args.max_evals}, obj_seeds={args.obj_seeds}, epochs={args.num_epochs}, "
          f"patience={args.patience}, coarse_dtype={args.coarse_dtype}", flush=True)

    for dataset in args.datasets:
        t0 = time.time()
        ds = load_hetero(dataset)
        type_order, node_counts = ds["type_order"], ds["node_counts"]
        coarse_counts = [max(1, int(round(args.ratio * n))) for n in node_counts]
        print(f"\n[clustering] === {dataset} === node_counts={node_counts} "
              f"coarse={coarse_counts}", flush=True)

        best_params, best_nmi = sweep(ds, type_order, node_counts, coarse_counts,
                                      args.max_evals, args.iters, device,
                                      args.num_epochs, args.patience, args.obj_seeds,
                                      coarse_dtype, he_init=args.he_init)
        print(f"[clustering] {dataset} best robust NMI={best_nmi*100:.2f}% "
              f"params={best_params}", flush=True)

        per_model_nmi = {m: [] for m in MODELS}
        per_model_ari = {m: [] for m in MODELS}
        empties = []
        for sd in range(args.n_seeds):
            result = run_befgc_hetero(
                ds["A_global"], ds["X_dict"], type_order, node_counts, coarse_counts,
                alpha=best_params["alpha"], beta=best_params["beta"],
                gamma=best_params["gamma"], eta=best_params["eta"],
                T=args.iters, seed=sd, device=device, dtype=coarse_dtype,
            )
            empties.append(result["empty_supernodes"])
            row = []
            for m in MODELS:
                torch.manual_seed(sd)
                nmi, ari = train_and_eval_clustering(ds, result, model_name=m,
                                                     num_epochs=args.num_epochs,
                                                     patience=args.patience, he_init=args.he_init)
                per_model_nmi[m].append(nmi)
                per_model_ari[m].append(ari)
                row.append(f"{m.replace('Hetero','')}: NMI={nmi*100:.1f} ARI={ari*100:.1f}")
            print(f"[clustering]   seed {sd}: " + " | ".join(row) +
                  f" empty={result['empty_supernodes']}", flush=True)

        rec = {"dataset": dataset, "ratio": args.ratio, "task": "node_clustering",
               "best_params": best_params, "avg_empty": float(np.mean(empties)),
               "n_seeds": args.n_seeds, "mins": (time.time() - t0) / 60,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"[clustering] === {dataset} RESULT (avg_empty={np.mean(empties):.1f}) ===", flush=True)
        for m in MODELS:
            a = np.array(per_model_nmi[m]) * 100
            r_ = np.array(per_model_ari[m]) * 100
            rec[m + "_nmi_mean"], rec[m + "_nmi_std"] = float(a.mean()), float(a.std())
            rec[m + "_ari_mean"], rec[m + "_ari_std"] = float(r_.mean()), float(r_.std())
            print(f"[clustering]   {m:11s}: NMI={a.mean():.2f}+/-{a.std():.2f}  "
                  f"ARI={r_.mean():.2f}+/-{r_.std():.2f}", flush=True)
        with open(args.out_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print("\n[clustering] ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
