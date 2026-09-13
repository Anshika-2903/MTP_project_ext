"""
Downstream task 1.1 -- Link prediction (ROC-AUC, AP) on top of BEFGC-hetero's
coarsened graphs.

Mirrors the EXISTING node-classification protocol (eval_hetero_fast.py) as
closely as possible so results are directly comparable in spirit:
  * TRAIN the encoder's weights via a supervised loss computed on the SMALL
    COARSENED graph (there: coarse pseudo-labels by majority vote; here:
    coarse "positive/negative edge" pairs read off the coarsened adjacency Ac).
  * EVALUATE by running the SAME weights through a forward pass on the
    ORIGINAL-size graph and scoring held-out original edges -- exactly the
    same "train small, evaluate at full scale" structure already used for
    node classification, just with an edge-scoring head instead of a softmax
    classification head.

Leakage handling: val/test positive edges of the target relation are removed
from BOTH the adjacency used to compute the coarsening (Ac/Xc_dict, done by
the caller before calling into this file) and the adjacency used for the
encoder's message passing at train/eval time (`A_masked`, built here). This
is a conservative (removes val edges from message passing too, not just
test), simple, clearly-leakage-free choice -- can be relaxed later to the
OGB-style two-graph (train-only vs train+val) convention if needed.

Reuses `_offsets`/`_coarse_adj_t_dict`/`_orig_adj_t_dict` from eval_hetero.py
and the `MODELS` classes from models_hetero.py UNMODIFIED (subclassed only to
expose embeddings instead of classification logits). Does not modify any
existing file.
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from befgc_hetero import _blocks
from eval_hetero import _offsets, _coarse_adj_t_dict, _orig_adj_t_dict
from eval_hetero_fast import _apply_he_init
from models_hetero import HeteroSGC, HeteroGCN, HeteroGCN2


# ---------------------------------------------------------------- encoders ---
# Same modules/parameters as the classification models (in_lin_dict/lins,
# alpha, beta, num_layers all set by the parent __init__); only forward()
# is overridden to stop before out_lin and return the per-type embedding
# dict instead of log_softmax(out_lin(h_dict[target])).

class HeteroSGCEncoder(HeteroSGC):
    def forward(self, x_dict, adj_t_dict):
        h_dict = {}
        for nt in x_dict:
            h_dict[nt] = self.in_lin_dict[nt][0](x_dict[nt]).relu_()
            for lin in self.in_lin_dict[nt][1:]:
                h_dict[nt] = lin(h_dict[nt]).relu_()
        for _ in range(self.num_layers):
            out_dict = {nt: [self.alpha * x] for nt, x in h_dict.items()}
            for et, adj_t in adj_t_dict.items():
                s, _, d = et
                out_dict[d].append(adj_t @ h_dict[s])
            for nt in x_dict:
                h_dict[nt] = torch.sum(torch.stack(out_dict[nt], dim=0), dim=0)
        return h_dict


class HeteroGCNEncoder(HeteroGCN):
    def forward(self, x_dict, adj_t_dict):
        h_dict = {}
        for l in range(self.num_layers):
            for nt in x_dict:
                src = x_dict[nt] if l == 0 else h_dict[nt]
                h_dict[nt] = F.relu_(self.lins[nt][l](src))
            out_dict = {nt: [self.alpha * x] for nt, x in h_dict.items()}
            for et, adj_t in adj_t_dict.items():
                s, _, d = et
                out_dict[d].append(adj_t @ h_dict[s])
            for nt in x_dict:
                h_dict[nt] = torch.mean(torch.stack(out_dict[nt], dim=0), dim=0)
        return h_dict


class HeteroGCN2Encoder(HeteroGCN2):
    def forward(self, x_dict, adj_t_dict):
        h_dict, h0_dict = {}, {}
        for l in range(self.num_layers):
            for nt in x_dict:
                if l == 0:
                    h_dict[nt] = F.relu_(self.lins[nt][l](x_dict[nt]))
                    h0_dict[nt] = h_dict[nt].detach()
                else:
                    h_dict[nt] = F.relu_(self.lins[nt][l](h_dict[nt]))
            out_dict = {nt: [self.alpha * x] for nt, x in h_dict.items()}
            for et, adj_t in adj_t_dict.items():
                s, _, d = et
                out_dict[d].append(adj_t @ h_dict[s])
            for nt in x_dict:
                agg = torch.mean(torch.stack(out_dict[nt], dim=0), dim=0)
                h_dict[nt] = self.beta * h0_dict[nt] + (1 - self.beta) * agg
        return h_dict


ENCODERS = {"HeteroSGC": HeteroSGCEncoder, "HeteroGCN": HeteroGCNEncoder,
            "HeteroGCN2": HeteroGCN2Encoder}


# --------------------------------------------------------- relation / split --

def pick_default_relation(dataset):
    """Pick the ORIGINAL (non-reversed) canonical relation with the most
    edges as the link-prediction target -- gives a meaningfully-sized task
    without requiring the caller to know the dataset's schema."""
    data = dataset["data"]
    best, best_n = None, -1
    for et in data.edge_types:
        n = data[et].edge_index.shape[1]
        if n > best_n:
            best, best_n = et, n
    return best


def sample_link_split_and_mask(dataset, rel=None, val_frac=0.1, test_frac=0.1, seed=0):
    """Split relation `rel`'s edges into train/val/test positives, sample
    matching negatives for val/test, and build a leakage-free adjacency
    (`A_masked`) with the held-out positives removed (both directions,
    since A_global is symmetrized)."""
    data = dataset["data"]
    type_order, node_counts = dataset["type_order"], dataset["node_counts"]
    tix = {t: i for i, t in enumerate(type_order)}
    fo = _blocks(node_counts)

    rel = rel or pick_default_relation(dataset)
    s, _, d = rel
    ei = data[rel].edge_index.numpy()
    src, dst = ei[0], ei[1]
    P = len(src)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(P)
    n_test = max(1, int(P * test_frac))
    n_val = max(1, int(P * val_frac))
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    def pack(idx):
        return np.stack([src[idx], dst[idx]], axis=1)

    train_pos, val_pos, test_pos = pack(train_idx), pack(val_idx), pack(test_idx)

    n_s, n_d = node_counts[tix[s]], node_counts[tix[d]]
    pos_set = set(map(tuple, np.stack([src, dst], axis=1).tolist()))

    def sample_neg(n, salt):
        rng2 = np.random.default_rng(seed * 7919 + salt)
        out, tries = [], 0
        while len(out) < n and tries < n * 200:
            a, b = int(rng2.integers(0, n_s)), int(rng2.integers(0, n_d))
            tries += 1
            if (a, b) not in pos_set:
                out.append((a, b))
        return np.array(out)

    val_neg = sample_neg(len(val_pos), 1)
    test_neg = sample_neg(len(test_pos), 2)

    A_masked = dataset["A_global"].tolil()
    i0, j0 = fo[tix[s]], fo[tix[d]]
    for a, b in np.concatenate([val_pos, test_pos], axis=0):
        A_masked[i0 + a, j0 + b] = 0.0
        A_masked[j0 + b, i0 + a] = 0.0
    A_masked = A_masked.tocsr()

    return {"rel": rel, "train_pos": train_pos, "val_pos": val_pos, "test_pos": test_pos,
            "val_neg": val_neg, "test_neg": test_neg, "A_masked": A_masked}


def _coarse_pos_neg(Ac, co, i, j, seed=0, thresh=1e-6):
    """Positive/negative supernode-pairs read off the COARSENED adjacency:
    an Ac entry above `thresh` means at least one member-node edge existed
    between those two supernodes."""
    block = Ac[co[i]:co[i + 1], co[j]:co[j + 1]]
    pos = (block > thresh).nonzero(as_tuple=False).cpu().numpy()
    m_s, m_d = block.shape
    pos_set = set(map(tuple, pos.tolist()))
    rng = np.random.default_rng(seed)
    neg, tries = [], 0
    while len(neg) < len(pos) and tries < max(1, len(pos)) * 200:
        a, b = int(rng.integers(0, m_s)), int(rng.integers(0, m_d))
        tries += 1
        if (a, b) not in pos_set:
            neg.append((a, b))
    return pos, np.array(neg)


# --------------------------------------------------------------- train/eval --

def train_and_eval_linkpred(
    dataset, result, split, model_name="HeteroGCN",
    hidden=64, num_layers=3, num_epochs=300, lr=0.01, weight_decay=5e-4,
    patience=40, he_init=False, dtype=torch.float32,
):
    """Same train-on-coarse/eval-on-original structure as
    eval_hetero_fast.downstream_accuracy_hetero_fast, with an edge-scoring
    (dot-product) head in place of the softmax classification head.
    Returns (test_roc_auc, test_ap)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    type_order, relations = dataset["type_order"], dataset["relations"]
    tix = {t: i for i, t in enumerate(type_order)}
    fo_fine = _offsets(dataset["node_counts"])
    co = result["coarse_offsets"]

    xc_dict = {t: result["Xc_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_c = _coarse_adj_t_dict(relations, result["Ac"].to(device), co, type_order, device, dtype)

    x_orig = {t: dataset["X_dict"][t].to(device=device, dtype=dtype) for t in type_order}
    adj_t_eval = _orig_adj_t_dict(relations, split["A_masked"], fo_fine, type_order, device, dtype)

    s, _, d = split["rel"]
    i, j = tix[s], tix[d]
    coarse_pos, coarse_neg = _coarse_pos_neg(result["Ac"], co, i, j, seed=0)
    if len(coarse_pos) == 0 or len(coarse_neg) == 0:
        raise RuntimeError(f"Not enough coarse {split['rel']} pairs to train link "
                            f"prediction (pos={len(coarse_pos)}, neg={len(coarse_neg)}); "
                            f"try a larger coarsening ratio.")

    metadata = (type_order, relations)
    model = ENCODERS[model_name](metadata, hidden, hidden, dataset["target_type"],
                                 num_layers=num_layers).to(device)
    model.eval()
    with torch.no_grad():
        model(xc_dict, adj_t_c)          # lazy Linear(-1,.) shape init
    if he_init:
        _apply_he_init(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_i = torch.tensor(coarse_pos[:, 0], device=device, dtype=torch.long)
    pos_j = torch.tensor(coarse_pos[:, 1], device=device, dtype=torch.long)
    neg_i = torch.tensor(coarse_neg[:, 0], device=device, dtype=torch.long)
    neg_j = torch.tensor(coarse_neg[:, 1], device=device, dtype=torch.long)

    val_pos, val_neg = split["val_pos"], split["val_neg"]

    def _scores(h_dict, idx_a, idx_b):
        return (h_dict[s][idx_a] * h_dict[d][idx_b]).sum(dim=-1)

    best_val_auc, best_weights, since_improve = -1.0, None, 0
    for _ in range(num_epochs):
        model.train()
        opt.zero_grad()
        h_c = model(xc_dict, adj_t_c)
        pos_score = _scores(h_c, pos_i, pos_j)
        neg_score = _scores(h_c, neg_i, neg_j)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            h_o = model(x_orig, adj_t_eval)
            vp = _scores(h_o, torch.tensor(val_pos[:, 0], device=device, dtype=torch.long),
                              torch.tensor(val_pos[:, 1], device=device, dtype=torch.long))
            vn = _scores(h_o, torch.tensor(val_neg[:, 0], device=device, dtype=torch.long),
                              torch.tensor(val_neg[:, 1], device=device, dtype=torch.long))
            vscores = torch.cat([vp, vn]).cpu().numpy()
            vlabels = np.concatenate([np.ones(len(vp)), np.zeros(len(vn))])
            val_auc = roc_auc_score(vlabels, vscores)
        if val_auc > best_val_auc:
            best_val_auc, best_weights, since_improve = val_auc, deepcopy(model.state_dict()), 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        h_o = model(x_orig, adj_t_eval)
        test_pos, test_neg = split["test_pos"], split["test_neg"]
        tp = _scores(h_o, torch.tensor(test_pos[:, 0], device=device, dtype=torch.long),
                          torch.tensor(test_pos[:, 1], device=device, dtype=torch.long))
        tn = _scores(h_o, torch.tensor(test_neg[:, 0], device=device, dtype=torch.long),
                          torch.tensor(test_neg[:, 1], device=device, dtype=torch.long))
        tscores = torch.cat([tp, tn]).cpu().numpy()
        tlabels = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])

    return roc_auc_score(tlabels, tscores), average_precision_score(tlabels, tscores)
