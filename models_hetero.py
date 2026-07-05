"""
Heterogeneous GNN backbones — ported VERBATIM from the official HGCond repo
(github.com/jianjianGJ/hgcond, ref [25]), which is what AH-UGC uses for its
heterogeneous node-classification benchmark. These replace the earlier LLM-
written reconstructions (which had dropout / no residual / wrong aggregation).

Key facts (from HGCond):
  * All three: per-type input Linear projection, residual term, NO dropout.
  * Message passing is `adj_t @ h` with `adj_t = asymmetric_gcn_norm(A)` per
    edge type (edge_type = (src, rel, dst); adj_t has shape (n_dst, n_src)).
  * HeteroSGC: SUM aggregation, alpha=0.01. HeteroGCN: MEAN, alpha=1.
    HeteroGCN2: GCNII-style skip (beta*h0 + (1-beta)*MEAN), alpha=1, beta=0.1.
  * Default architecture: num_layers=3, hidden=64.

We keep HGCond's forward exactly, adapt only the constructor signature to our
harness — (metadata, hidden, out, target) — and append log_softmax so our
existing F.nll_loss training loop works unchanged.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_sparse import SparseTensor, sum as sparsesum, mul


def asymmetric_gcn_norm(adj_t):
    """GCN-style symmetric-ish normalization for a (possibly rectangular)
    bipartite adjacency: D_dst^-1/2 A D_src^-1/2. Verbatim from HGCond utils."""
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.0)
    deg_src = sparsesum(adj_t, dim=0) + 1e-5
    deg_src_inv_sqrt = deg_src.pow_(-0.5)
    deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt == float("inf"), 0.0)
    deg_dst = sparsesum(adj_t, dim=1) + 1e-5
    deg_dst_inv_sqrt = deg_dst.pow_(-0.5)
    deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt == float("inf"), 0.0)
    adj_t = mul(adj_t, deg_dst_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_src_inv_sqrt.view(1, -1))
    return adj_t


class HeteroSGC(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, target,
                 num_layers=3, num_lins=1, alpha=0.01):
        super().__init__()
        node_types, _ = metadata
        self.alpha, self.num_layers, self.target = alpha, num_layers, target
        self.in_lin_dict = torch.nn.ModuleDict()
        for nt in node_types:
            self.in_lin_dict[nt] = torch.nn.ModuleList()
            self.in_lin_dict[nt].append(Linear(-1, hidden_channels))
            for _ in range(num_lins - 1):
                self.in_lin_dict[nt].append(Linear(hidden_channels, hidden_channels))
        self.out_lin = Linear(hidden_channels, out_channels)

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
        return F.log_softmax(self.out_lin(h_dict[self.target]), dim=1)


class HeteroGCN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, target,
                 num_layers=3, alpha=1):
        super().__init__()
        node_types, _ = metadata
        self.alpha, self.num_layers, self.target = alpha, num_layers, target
        self.lins = torch.nn.ModuleDict()
        for nt in node_types:
            self.lins[nt] = torch.nn.ModuleList()
            self.lins[nt].append(Linear(-1, hidden_channels, bias=False))
            for _ in range(num_layers - 1):
                self.lins[nt].append(Linear(hidden_channels, hidden_channels, bias=False))
        self.out_lin = Linear(hidden_channels, out_channels)

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
        return F.log_softmax(self.out_lin(h_dict[self.target]), dim=1)


class HeteroGCN2(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, target,
                 num_layers=3, alpha=1, beta=0.1):
        super().__init__()
        node_types, _ = metadata
        self.alpha, self.beta = alpha, beta
        self.num_layers, self.target = num_layers, target
        self.lins = torch.nn.ModuleDict()
        for nt in node_types:
            self.lins[nt] = torch.nn.ModuleList()
            self.lins[nt].append(Linear(-1, hidden_channels, bias=False))
            for _ in range(num_layers - 1):
                self.lins[nt].append(Linear(hidden_channels, hidden_channels, bias=False))
        self.out_lin = Linear(hidden_channels, out_channels)

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
        return F.log_softmax(self.out_lin(h_dict[self.target]), dim=1)


MODELS = {"HeteroSGC": HeteroSGC, "HeteroGCN": HeteroGCN, "HeteroGCN2": HeteroGCN2}
