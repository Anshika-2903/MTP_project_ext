"""
Random-search hyperparameter sweep for BEFGC-hetero on a heterogeneous dataset.

The tuned single run showed the objective can DIVERGE (J increasing) at eta=0.3
with strong beta, so this sweep searches lower eta and tracks whether J actually
descended -- an unstable (diverging) trial is reported but flagged.
Evaluates one hetero backbone per trial (HeteroGCN by default) for speed.
"""
import argparse
import json
import time
import numpy as np
import torch

from data_loading_hetero import load_hetero
from befgc_hetero import run_befgc_hetero
from eval_hetero import downstream_accuracy_hetero

LOG_PATH = "hetero_sweep.jsonl"


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_params(rng):
    # eta capped well below 0.3 -- the failure mode was overshoot/divergence.
    return {
        "alpha": float(rng.uniform(0.0, 1.0)),
        "beta": loguniform(rng, 0.5, 10.0),
        "gamma": loguniform(rng, 1e-4, 0.1),
        "eta": loguniform(rng, 0.02, 0.25),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="IMDB")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--max_evals", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--model", default="HeteroGCN")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    ds = load_hetero(args.dataset)
    type_order, node_counts = ds["type_order"], ds["node_counts"]
    coarse_counts = [max(1, int(round(args.ratio * n))) for n in node_counts]
    M = sum(coarse_counts)
    print(f"[hsweep] {args.dataset}: node_counts={node_counts} coarse={coarse_counts} "
          f"(M={M}), trials={args.max_evals}, model={args.model}, device={device}", flush=True)

    rng = np.random.default_rng(args.seed)
    best_acc, best_params = -np.inf, None

    for t in range(args.max_evals):
        params = sample_params(rng)
        result = run_befgc_hetero(
            ds["A_global"], ds["X_dict"], type_order, node_counts, coarse_counts,
            alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
            eta=params["eta"], T=args.iters, seed=0, device=device,
        )
        acc = downstream_accuracy_hetero(ds, result, model_name=args.model)
        hist = result["history"]
        diverged = hist[-1] > hist[0] + 1e-6
        empty = result["empty_supernodes"]

        record = {
            "dataset": args.dataset, "ratio": args.ratio, "M": M,
            **params, "accuracy": acc, "empty_supernodes": empty,
            "J_start": hist[0], "J_end": hist[-1], "diverged": diverged,
            "model": args.model, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        tag = ""
        if acc > best_acc and not diverged:
            best_acc, best_params = acc, params
            tag = "  <-- new best"
        flag = " [DIVERGED]" if diverged else ""
        print(f"[{t+1:02d}/{args.max_evals}] acc={acc*100:5.2f}% | "
              f"alpha={params['alpha']:.3f} beta={params['beta']:.3f} "
              f"gamma={params['gamma']:.4f} eta={params['eta']:.4f} | "
              f"empty={empty}/{M}{flag}{tag}", flush=True)

    print(f"\n[hsweep] DONE. best (stable) acc={best_acc*100:.2f}%  params={best_params}")


if __name__ == "__main__":
    main()
