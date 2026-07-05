"""
Full heterogeneous pipeline for one or more datasets, in a single process
(meant to run as a PBS batch job on a GPU node):
  for each dataset:
    1. sweep hyperparameters (tuned to a chosen model, default HeteroSGC)
    2. multi-seed the best config across ALL three backbones
    3. write the Table-4-style result (mean +/- std per model) to hetero_final.jsonl

Uses GPU automatically (device=cuda) when available.
"""
import argparse
import json
import time
import numpy as np
import torch

from data_loading_hetero import load_hetero
from befgc_hetero import run_befgc_hetero
from eval_hetero import downstream_accuracy_hetero
from sweep_hetero import sample_params

MODELS = ["HeteroSGC", "HeteroGCN", "HeteroGCN2"]


def sweep(ds, type_order, node_counts, coarse_counts, max_evals, iters,
          tune_model, device, objective="mean"):
    """Select coarsening hyperparameters (alpha,beta,gamma,eta).

    objective="sgc"  -> tune on HeteroSGC only (old behaviour).
    objective="mean" -> tune on the MEAN accuracy across all three backbones,
                        so a config that collapses 2/3 backbones (as the
                        SGC-only objective allowed on IMDB) can never win.
    """
    rng = np.random.default_rng(0)
    best_score, best_params = -np.inf, None
    for t in range(max_evals):
        params = sample_params(rng)
        result = run_befgc_hetero(
            ds["A_global"], ds["X_dict"], type_order, node_counts, coarse_counts,
            alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
            eta=params["eta"], T=iters, seed=0, device=device,
        )
        diverged = result["history"][-1] > result["history"][0] + 1e-6
        # Seed the eval so the sweep score is REPRODUCIBLE and representative of
        # the multi-seed phase (previously the model init used leftover RNG
        # state, so a config could look healthy on one lucky init but collapse
        # across the fixed multi-seed inits).
        if objective == "sgc":
            torch.manual_seed(0)
            accs = {tune_model: downstream_accuracy_hetero(ds, result, model_name=tune_model)}
        else:
            accs = {}
            for m in MODELS:
                torch.manual_seed(0)
                accs[m] = downstream_accuracy_hetero(ds, result, model_name=m)
        score = float(np.mean(list(accs.values())))
        tag = ""
        if score > best_score and not diverged:
            best_score, best_params = score, params
            tag = "  <-- best"
        accstr = " ".join(f"{m.replace('Hetero','')}={a*100:.1f}" for m, a in accs.items())
        print(f"  [sweep {t+1:02d}/{max_evals}] {accstr} | score={score*100:5.2f}% "
              f"empty={result['empty_supernodes']} "
              f"(a={params['alpha']:.2f} b={params['beta']:.2f} "
              f"g={params['gamma']:.4f} e={params['eta']:.3f}){' DIVERGED' if diverged else ''}{tag}",
              flush=True)
    return best_params, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ACM", "DBLP"])
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--max_evals", type=int, default=25)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--tune_model", default="HeteroSGC")
    p.add_argument("--objective", default="mean", choices=["mean", "sgc"],
                   help="'mean' tunes on avg of all 3 backbones (avoids collapse); "
                        "'sgc' tunes on HeteroSGC only (old behaviour)")
    p.add_argument("--out_jsonl", default="hetero_final.jsonl",
                   help="output path for the per-dataset result records")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[full] device={device}, datasets={args.datasets}, ratio={args.ratio}, "
          f"sweep={args.max_evals}, seeds={args.n_seeds}", flush=True)

    for dataset in args.datasets:
        t0 = time.time()
        ds = load_hetero(dataset)
        type_order, node_counts = ds["type_order"], ds["node_counts"]
        coarse_counts = [max(1, int(round(args.ratio * n))) for n in node_counts]
        print(f"\n[full] === {dataset} === node_counts={node_counts} "
              f"coarse={coarse_counts}", flush=True)

        best_params, best_acc = sweep(ds, type_order, node_counts, coarse_counts,
                                      args.max_evals, args.iters, args.tune_model, device,
                                      objective=args.objective)
        print(f"[full] {dataset} best score({args.objective})={best_acc*100:.2f}% "
              f"params={best_params}", flush=True)

        # multi-seed the best config across all three backbones
        per_model = {m: [] for m in MODELS}
        empties = []
        for s in range(args.n_seeds):
            result = run_befgc_hetero(
                ds["A_global"], ds["X_dict"], type_order, node_counts, coarse_counts,
                alpha=best_params["alpha"], beta=best_params["beta"],
                gamma=best_params["gamma"], eta=best_params["eta"],
                T=args.iters, seed=s, device=device,
            )
            empties.append(result["empty_supernodes"])
            for m in MODELS:
                torch.manual_seed(s)
                per_model[m].append(downstream_accuracy_hetero(ds, result, model_name=m))

        rec = {"dataset": dataset, "ratio": args.ratio,
               "best_params": best_params, "avg_empty": float(np.mean(empties)),
               "n_seeds": args.n_seeds, "mins": (time.time() - t0) / 60,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"[full] === {dataset} RESULT (avg_empty={np.mean(empties):.1f}) ===", flush=True)
        for m in MODELS:
            a = np.array(per_model[m]) * 100
            rec[m + "_mean"], rec[m + "_std"] = float(a.mean()), float(a.std())
            print(f"[full]   {m:11s}: {a.mean():.2f} +/- {a.std():.2f}", flush=True)
        with open(args.out_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print("\n[full] ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
