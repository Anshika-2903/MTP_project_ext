import argparse
import torch

from data_loading_hetero import load_hetero
from befgc_hetero import run_befgc_hetero
from eval_hetero import downstream_accuracy_hetero


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="IMDB")
    p.add_argument("--ratio", type=float, default=0.3)   # AH-UGC Table 4 uses 30%
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--models", nargs="+", default=["HeteroSGC", "HeteroGCN", "HeteroGCN2"])
    args = p.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[main-h] device={device}")

    ds = load_hetero(args.dataset)
    type_order, node_counts = ds["type_order"], ds["node_counts"]
    # per-type target sizes at the given ratio (>=1 supernode each)
    coarse_counts = [max(1, int(round(args.ratio * n))) for n in node_counts]

    print(f"[main-h] {args.dataset}: types={type_order} node_counts={node_counts} "
          f"-> coarse_counts={coarse_counts} (ratio={args.ratio}) relations={ds['relations']}")

    result = run_befgc_hetero(
        ds["A_global"], ds["X_dict"], type_order, node_counts, coarse_counts,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        eta=args.eta, T=args.iters, seed=args.seed, verbose=True, dtype=dtype, device=device,
    )
    print(f"[main-h] empty supernodes: {result['empty_supernodes']} / {sum(coarse_counts)}")

    for mname in args.models:
        torch.manual_seed(args.seed)
        # eval GNN runs in float32 (HGCond default); coarsening dtype is separate
        acc = downstream_accuracy_hetero(ds, result, model_name=mname)
        print(f"[main-h] {mname}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
