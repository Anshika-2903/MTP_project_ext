"""
Downstream task 1.2 -- Node clustering (NMI, ARI) on top of BEFGC-hetero's
coarsened graphs.

Protocol: train the SAME encoder architecture used for link prediction
(link_pred_hetero.py's HeteroSGCEncoder/HeteroGCNEncoder/HeteroGCN2Encoder,
reused UNMODIFIED) plus a small trainable linear classification head, on the
COARSE graph via the exact same coarse-pseudo-label-by-majority-vote
node-classification objective already used by eval_hetero_fast.py. This
keeps training identical in spirit to node classification -- the only
difference is what we do with the trained model afterward.

At evaluation time, run the trained encoder on the ORIGINAL (uncoarsened)
graph to get per-node embeddings for the target type, and cluster them with
k-means (k = number of classes) -- this step is unsupervised, k-means never
sees the true labels. The true labels of the test nodes are used ONLY
afterward, to SCORE the resulting clustering via NMI and ARI (how well the
unsupervised clusters line up with the real classes) -- never to influence
training or the clustering itself. This is the standard "does the learned
embedding space separate classes well" evaluation used for heterogeneous
graph embeddings (e.g. in HAN/metapath2vec-style papers).

Same train-on-coarse/eval-on-original structure as node classification
(eval_hetero_fast.py) and link prediction (link_pred_hetero.py) -- all three
downstream tasks share this methodology for consistency. Does not modify
any of those files.
"""
import torch
import torch.nn.functional as F
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from eval_hetero import _offsets, _coarse_adj_t_dict, _orig_adj_t_dict
from eval_hetero_fast import _apply_he_init
from link_pred_hetero import ENCODERS


class _EncoderWithHead(torch.nn.Module):
    """Wraps an embedding-only encoder (from link_pred_hetero.ENCODERS) with
    a trainable linear classification head, so the same backbone used for
    link prediction can also be trained via a supervised objective. Needed
    because raw, untrained coarse features alone are a weak clustering
    baseline -- the head gives the encoder a training signal, but only the
    ENCODER's output embeddings (not the head's logits) are what actually
    gets clustered at evaluation time."""

    def __init__(self, encoder, hidden, num_classes, target):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(hidden, num_classes)
        self.target = target

    def forward(self, x_dict, adj_t_dict):
        h_dict = self.encoder(x_dict, adj_t_dict)
        logits = F.log_softmax(self.head(h_dict[self.target]), dim=1)
        return logits, h_dict


def train_and_eval_clustering(
    dataset, result, model_name="HeteroGCN",
    hidden=64, num_layers=3, num_epochs=300, lr=0.01, weight_decay=5e-4,
    patience=40, he_init=False, dtype=torch.float32, n_clusters=None,
):
    """Train-on-coarse (majority-vote pseudo-labels) / eval-on-original,
    identical protocol to eval_hetero_fast.downstream_accuracy_hetero_fast,
    but scores the ORIGINAL graph's target-type embeddings via k-means
    clustering (NMI, ARI) instead of classification accuracy.
    Returns (nmi, ari)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset["data"]
    type_order = dataset["type_order"]
    target = dataset["target_type"]
    relations = dataset["relations"]
    fo_fine = _offsets(dataset["node_counts"])

    tix = {t: i for i, t in enumerate(type_order)}
    co = result["coarse_offsets"]
    C_hard = result["C_hard"]

    xc_dict = {t: result["Xc_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_c = _coarse_adj_t_dict(relations, result["Ac"].to(device), co, type_order, device, dtype)

    labels = data[target].y.to(device)
    valid = labels >= 0
    num_classes = int(labels[valid].max().item()) + 1
    k = n_clusters or num_classes
    ti = tix[target]
    fo = result["fine_offsets"]
    C_tgt = C_hard[fo[ti]:fo[ti + 1], co[ti]:co[ti + 1]].to(device=device, dtype=torch.float32)
    onehot = F.one_hot(labels.clamp(min=0), num_classes).to(torch.float32)
    onehot[~valid] = 0
    class_counts = C_tgt.t() @ onehot
    labels_c = class_counts.argmax(dim=1)
    nonempty = class_counts.sum(dim=1) > 0

    x_orig = {t: dataset["X_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_o = _orig_adj_t_dict(relations, dataset["A_global"], fo_fine, type_order, device, dtype)
    val_mask = data[target].val_mask.to(device) if hasattr(data[target], "val_mask") else None
    if val_mask is not None:
        val_mask = val_mask & valid

    metadata = (type_order, relations)
    model = _EncoderWithHead(
        ENCODERS[model_name](metadata, hidden, hidden, target, num_layers=num_layers),
        hidden, num_classes, target,
    ).to(device)
    model.eval()
    with torch.no_grad():
        model(xc_dict, adj_t_c)          # lazy Linear(-1,.) shape init
    if he_init:
        _apply_he_init(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc, best_weights, since_improve = -1.0, None, 0
    for _ in range(num_epochs):
        model.train()
        opt.zero_grad()
        out, _ = model(xc_dict, adj_t_c)
        loss = F.nll_loss(out[nonempty], labels_c[nonempty])
        loss.backward()
        opt.step()

        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                pred_val, _ = model(x_orig, adj_t_o)
                pred_val = pred_val.argmax(dim=-1)
                val_acc = (pred_val[val_mask] == labels[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc, best_weights, since_improve = val_acc, deepcopy(model.state_dict()), 0
            else:
                since_improve += 1
                if since_improve >= patience:
                    break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        _, h_dict = model(x_orig, adj_t_o)
        emb = h_dict[target].detach().cpu().numpy()

    test_mask = data[target].test_mask.to(device) if hasattr(data[target], "test_mask") else None
    m = (test_mask & valid) if (test_mask is not None and test_mask.sum() > 0) else valid
    m_np = m.cpu().numpy()
    true_labels = labels.cpu().numpy()[m_np]

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb[m_np])
    pred_clusters = km.labels_

    nmi = normalized_mutual_info_score(true_labels, pred_clusters)
    ari = adjusted_rand_score(true_labels, pred_clusters)
    return nmi, ari
