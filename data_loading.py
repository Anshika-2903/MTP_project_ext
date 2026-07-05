import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

# Planetoid dataset names use this exact casing.
_NAME_MAP = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def load_data(name="Cora", data_dir="./data"):
    """Load a homophilous Planetoid dataset as dense tensors.

    Returns X (n x d), A (n x n dense adjacency, 0/1, symmetric), labels (n,), num_classes.
    """
    canonical = _NAME_MAP.get(name.lower(), name)
    root = os.path.join(data_dir, canonical)
    os.makedirs(root, exist_ok=True)

    dataset = Planetoid(root=data_dir, name=canonical)
    data = dataset[0]

    A = to_dense_adj(data.edge_index)[0]
    X = data.x.float()
    labels = data.y.long()
    num_classes = dataset.num_classes

    # L2 row-normalize features. Cora/CiteSeer bag-of-words vectors have
    # wildly varying L1 norms; without this, squared-Euclidean feature
    # distances dwarf the structural/balance/entropy terms in the BEFGC
    # objective and the mirror step saturates to a near-hard assignment on
    # the very first iteration (same normalization step used in
    # fgc_hetero2/training.py after each feature update).
    norms = X.norm(p=2, dim=1, keepdim=True)
    norms[norms == 0] = 1
    X = X / norms

    return X, A, labels, num_classes
