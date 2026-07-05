"""
Multi-seed IMDB benchmark for a fixed BEFGC-hetero config, evaluated across all
three AH-UGC backbones (HeteroSGC, HeteroGCN, HeteroGCN2). Reports mean +/- std
per model, matching AH-UGC Table 4. For each seed the coarsening is run once and
all three models are trained/evaluated on that same coarsened graph.
"""
import argparse
import json
import time
import numpy as np
import torch

from data_loading_hetero import load_hetero
from befgc_hetero import run_befgc_hetero
from eval_hetero import downstream_accuracy_hetero

MODELS = ["HeteroSGC", "HeteroGCN", "HeteroGCN2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="IMDB")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.36511)
    p.add_argument("--beta", type=float, default=0.68584)
    p.add_argument("--gamma", type=float, default=0.007715)
    p.add_argument("--eta", type=float, default=0.20799)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[bench-h] device={device}", flush=True)

    ds = load_hetero(args.dataset)
    type_order, node_counts = ds["type_order"], ds["node_counts"]
    coarse_counts = [max(1, int(round(args.ratio * n))) for n in node_counts]
    M = sum(coarse_counts)
    print(f"[bench-h] {args.dataset} r={args.ratio}: node_counts={node_counts} "
          f"coarse={coarse_counts} (M={M})", flush=True)
    print(f"[bench-h] params alpha={args.alpha} beta={args.beta} "
          f"gamma={args.gamma} eta={args.eta}, seeds={args.n_seeds}", flush=True)

    per_model = {m: [] for m in MODELS}
    empties = []
    for s in range(args.n_seeds):
        result = run_befgc_hetero(
            ds["A_global"], ds["X_dict"], type_order, node_counts, coarse_counts,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            eta=args.eta, T=args.iters, seed=s, device=device,
        )
        empties.append(result["empty_supernodes"])
        line = [f"seed {s} (empty={result['empty_supernodes']}/{M}):"]
        for mname in MODELS:
            torch.manual_seed(s)
            acc = downstream_accuracy_hetero(ds, result, model_name=mname)
            per_model[mname].append(acc)
            line.append(f"{mname}={acc*100:.2f}%")
        print("  " + "  ".join(line), flush=True)

    print(f"\n[bench-h] === {args.dataset} r={args.ratio} SUMMARY "
          f"(avg empty={np.mean(empties):.1f}/{M}) ===")
    summary = {"dataset": args.dataset, "ratio": args.ratio, "M": M,
               "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma, "eta": args.eta,
               "n_seeds": args.n_seeds, "avg_empty": float(np.mean(empties)),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    for mname in MODELS:
        a = np.array(per_model[mname]) * 100
        print(f"  {mname:12s}: {a.mean():.2f} +/- {a.std():.2f}")
        summary[mname + "_mean"] = float(a.mean())
        summary[mname + "_std"] = float(a.std())
        summary[mname + "_per_seed"] = a.tolist()
    with open("hetero_final.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
