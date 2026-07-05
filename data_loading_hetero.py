"""
Heterogeneous dataset loading for BEFGC-hetero (IMDB, DBLP, ACM).

Generalized over arbitrary PyG HeteroData: builds a single symmetric block
adjacency A_global from ALL edge types present, a per-type feature dict (with a
constant-feature fallback for typeless node types), and the relation list used
by the hetero GNN eval. Node types are NOT assumed to form a bipartite star and
the target type need not connect to every other type (DBLP: author only touches
paper), so we read the actual edge types from the data.

Returns a dict with:
  A_global    : (N x N) scipy CSR, symmetric block adjacency over all types
  X_dict      : {type: (n_t x d_t) float tensor}
  type_order  : list of node types (stacking order in A_global)
  node_counts : [n_t for t in type_order]
  relations   : list of (src,'to',dst) canonical edge types + reverses (for GNN)
  data        : raw PyG HeteroData (labels/edges at eval time)
  target_type : node type carrying labels
"""
import os
import shutil
import numpy as np
import scipy.sparse as sp
import torch


def _to_scipy(edge_index, num_rows, num_cols):
    row, col = edge_index.cpu().numpy()
    vals = np.ones(len(row))
    return sp.coo_matrix((vals, (row, col)), shape=(num_rows, num_cols)).tocsr()


def _raw_features(data, ntype):
    """Raw per-type features, or None if the type has no features."""
    store = data[ntype]
    if "x" in store and store.x is not None:
        x = store.x
        return (x.to_dense() if x.is_sparse else x).float()
    return None


def _l2norm(x):
    """L2 row-normalize. Prevents any one type's raw magnitude from dominating
    the feature-cost term (large magnitudes previously overflowed -> J=nan)."""
    n = x.norm(p=2, dim=1, keepdim=True)
    n[n == 0] = 1.0
    return x / n


def build_from_hetero_data(data, type_order, target_type):
    node_counts = [data[t].num_nodes for t in type_order]
    tix = {t: i for i, t in enumerate(type_order)}
    raw = {t: _raw_features(data, t) for t in type_order}

    # Assemble symmetric block adjacency from every edge type in the graph.
    blocks = [[sp.csr_matrix((node_counts[i], node_counts[j]))
               for j in range(len(type_order))] for i in range(len(type_order))]
    canonical = []
    for (s, rel, d) in data.edge_types:
        if s not in tix or d not in tix:
            continue
        i, j = tix[s], tix[d]
        A_ij = _to_scipy(data[(s, rel, d)].edge_index, node_counts[i], node_counts[j])
        blocks[i][j] = blocks[i][j] + A_ij
        blocks[j][i] = blocks[j][i] + A_ij.T          # symmetrize
        canonical.append((s, d))

    # ---- Per-type features ----
    # Featured types keep their own features. FEATURELESS types (ACM author/
    # subject, DBLP conference) get features by PROPAGATION: the mean of their
    # neighbours' features (from a connected featured type). This replaces the
    # old constant-1 fallback, which injected pure noise into message passing
    # and was the main cause of HeteroGCN collapsing (over-smoothing).
    X_dict = {}
    for i, t in enumerate(type_order):
        if raw[t] is not None:
            X_dict[t] = _l2norm(raw[t])
            continue
        prop = None
        for j, s in enumerate(type_order):
            if raw[s] is not None and blocks[i][j].nnz > 0:
                deg = np.asarray(blocks[i][j].sum(1)).flatten()
                deg[deg == 0] = 1.0
                agg = (blocks[i][j] @ raw[s].numpy()) / deg[:, None]   # mean of neighbour feats
                prop = torch.tensor(agg).float()
                break
        if prop is None:                              # no featured neighbour -> constant fallback
            prop = torch.ones(node_counts[i], 1)
        X_dict[t] = _l2norm(prop)

    A_global = sp.bmat(blocks).tocsr()
    A_global.data[:] = 1.0  # collapse any double-counted parallel edges to 0/1

    # Relations for the hetero GNN: both directions of each canonical edge.
    rel_set = []
    for (s, d) in canonical:
        for r in [(s, "to", d), (d, "to", s)]:
            if r not in rel_set:
                rel_set.append(r)

    return {
        "A_global": A_global,
        "X_dict": X_dict,
        "type_order": type_order,
        "node_counts": node_counts,
        "relations": rel_set,
        "data": data,
        "target_type": target_type,
    }


# ---------------------------------------------------------------- IMDB --------
def _fix_imdb_directories(root="./data/IMDB"):
    processed_dir, raw_dir = os.path.join(root, "processed"), os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_files = ["adjM.npz", "features_0.npz", "features_1.npz", "features_2.npz",
                 "labels.npy", "node_types.npy", "train_val_test_idx.npz"]
    if os.path.exists(processed_dir):
        for f in os.listdir(processed_dir):
            if f in raw_files:
                shutil.move(os.path.join(processed_dir, f), os.path.join(raw_dir, f))
        if os.path.exists(os.path.join(processed_dir, "data.pt")):
            os.remove(os.path.join(processed_dir, "data.pt"))


def load_imdb():
    from torch_geometric.datasets import IMDB
    _fix_imdb_directories()
    data = IMDB(root="./data/IMDB")[0]
    return build_from_hetero_data(data, ["movie", "actor", "director"], "movie")


# ---------------------------------------------------------------- DBLP --------
def load_dblp():
    from torch_geometric.datasets import DBLP
    data = DBLP(root="./data/DBLP")[0]
    # author(feat 334), paper(4231), term(50), conference(no features)
    types = [t for t in ["author", "paper", "term", "conference"] if t in data.node_types]
    return build_from_hetero_data(data, types, "author")


# ---------------------------------------------------------------- ACM ---------
def load_acm(path="./data/ACM.mat"):
    """Standard HAN/DGL ACM (ACM3025): 3025 papers in 3 classes (Database,
    Wireless, Data-Mining), node types paper/author/subject, target=paper.

    The PyG HGBDataset download is blocked by the IITD proxy, so we read the
    DGL ACM.mat (fetched on the laptop and scp'd to PRAGYA) and build a
    HeteroData with the canonical conference->label mapping + 20/10/70 split.
    """
    import numpy as np
    import scipy.io as sio
    from torch_geometric.data import HeteroData

    m = sio.loadmat(path)
    p_vs_a = m["PvsA"]           # paper-author
    p_vs_l = m["PvsL"]           # paper-subject(field)
    p_vs_t = m["PvsT"]           # paper-term (features)
    p_vs_c = m["PvsC"]           # paper-conference

    # Papers from these conferences form the 3-class task (canonical HAN setup).
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    p_sel = np.nonzero(p_vs_c[:, conf_ids].sum(1).A1)[0]
    p_vs_a, p_vs_l, p_vs_t, p_vs_c = (p_vs_a[p_sel], p_vs_l[p_sel],
                                      p_vs_t[p_sel], p_vs_c[p_sel])

    labels = np.zeros(len(p_sel), dtype=np.int64)
    pc_p, pc_c = p_vs_c.nonzero()
    for cid, lid in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == cid]] = lid

    data = HeteroData()
    data["paper"].x = torch.FloatTensor(p_vs_t.toarray())
    data["paper"].y = torch.LongTensor(labels)
    data["author"].num_nodes = p_vs_a.shape[1]
    data["subject"].num_nodes = p_vs_l.shape[1]

    pa = np.vstack(p_vs_a.nonzero())
    pl = np.vstack(p_vs_l.nonzero())
    data["paper", "to", "author"].edge_index = torch.LongTensor(pa)
    data["paper", "to", "subject"].edge_index = torch.LongTensor(pl)

    # 20/10/70 train/val/test split, stratified by conference (canonical).
    rng = np.random.default_rng(0)
    fmask = np.zeros(len(pc_p))
    for cid in conf_ids:
        idx = pc_c == cid
        fmask[idx] = rng.permutation(np.linspace(0, 1, int(idx.sum())))
    n = len(p_sel)
    train = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    train[pc_p[fmask <= 0.2]] = True
    test[pc_p[fmask > 0.3]] = True
    data["paper"].train_mask = train
    data["paper"].test_mask = test

    return build_from_hetero_data(data, ["paper", "author", "subject"], "paper")


LOADERS = {"IMDB": load_imdb, "DBLP": load_dblp, "ACM": load_acm}


def load_hetero(name="IMDB"):
    if name not in LOADERS:
        raise ValueError(f"Unknown hetero dataset {name}; available: {list(LOADERS)}")
    return LOADERS[name]()
