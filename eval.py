"""
Evaluation for BEFGC coarsenings.

Two kinds of metrics:
  1. Structural/feature preservation (no training): relative reconstruction
     error of the coarsened graph and features, plus the balance ratio.
  2. Downstream task quality: train a small GCN on the coarsened graph,
     evaluate on the original graph after lifting via pinv(C_hard), mirroring
     the protocol in fgc_hetero2/training.py::get_accuracy for comparability.
"""
import torch
import torch.nn.functional as F


class GCN(torch.nn.Module):
    """Same architecture as fgc_hetero2/models.py::Net, kept local so this
    project has no import dependency on the other repo."""

    def __init__(self, input_dim, num_classes, hidden=64):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def structural_metrics(A, X, result):
    """Cheap, training-free diagnostics for a BEFGC run."""
    Ac, Xc, nu = result["Ac"], result["Xc"], result["nu"]
    C_hard = result["C_hard"]
    m = Ac.shape[0]

    # Lift the coarsened graph/features back to n nodes via the hard
    # assignment and compare against the originals.
    A_lift = C_hard @ Ac @ C_hard.t()
    X_lift = C_hard @ Xc

    A = A.to(A_lift.dtype)
    X = X.to(X_lift.dtype)
    rel_A_err = (torch.norm(A_lift - A) / torch.norm(A).clamp_min(1e-12)).item()
    rel_X_err = (torch.norm(X_lift - X) / torch.norm(X).clamp_min(1e-12)).item()

    balance_ratio = (nu.min() / nu.max()).item()  # 1.0 = perfectly balanced
    empty_supernodes = int((C_hard.sum(dim=0) == 0).sum().item())

    return {
        "rel_adj_recon_error": rel_A_err,
        "rel_feat_recon_error": rel_X_err,
        "balance_ratio": balance_ratio,
        "empty_supernodes": empty_supernodes,
        "num_supernodes": m,
    }


def downstream_accuracy(A, X, labels, num_classes, result, num_epochs=100, lr=0.01, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C_hard = result["C_hard"].to(device)
    Ac = result["Ac"].to(device)
    Xc = result["Xc"].to(device)

    # Coarsened graph -> weighted sparse edge_index. Ac carries mass-weighted
    # connectivity, so we keep the weights (the old fgc_hetero2 eval discarded
    # them); GCNConv uses them for normalized propagation.
    Wc = Ac.clone()
    Wc.fill_diagonal_(0)
    Wc[Wc < 1e-6] = 0
    edge_index_c = Wc.nonzero(as_tuple=False).t().contiguous()
    edge_weight_c = Wc[edge_index_c[0], edge_index_c[1]]

    X_gpu = X.to(device=device, dtype=Xc.dtype)
    A_gpu = A.to(device)
    labels_gpu = labels.to(device).long()

    # Coarse pseudo-labels via majority vote per supernode (vectorized).
    # C_hard is one-hot, so C_hard^T @ onehot(labels) counts, per supernode,
    # how many of its members carry each class; argmax gives the majority.
    onehot = F.one_hot(labels_gpu, num_classes).to(Xc.dtype)   # (n, K)
    class_counts = C_hard.t() @ onehot                          # (m, K)
    labels_c = class_counts.argmax(dim=1)
    # Empty supernodes have all-zero counts -> argmax=0; mask them out of the
    # training loss so their meaningless (0-feature) row doesn't add noise.
    nonempty = C_hard.sum(dim=0) > 0

    model = GCN(X.shape[1], num_classes).to(device=device, dtype=Xc.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        out = model(Xc, edge_index_c, edge_weight_c)
        loss = F.nll_loss(out[nonempty], labels_c[nonempty])
        loss.backward()
        optimizer.step()

    model.eval()
    edge_index_orig = A_gpu.nonzero(as_tuple=False).t().contiguous()
    with torch.no_grad():
        pred = model(X_gpu, edge_index_orig).argmax(dim=1)
        acc = (pred == labels_gpu).float().mean().item()

    return acc
