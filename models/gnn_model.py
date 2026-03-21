"""GNN レコメンドモデル: GAT + APPNP エンコーダー, BPR Loss"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class HeteroGATEncoder(nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels=128, out_channels=64, num_layers=3, heads=4, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers; self.dropout = dropout
        self.projections = nn.ModuleDict({nt: Linear(dim, hidden_channels) for nt, dim in in_channels_dict.items()})
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        for layer_i in range(num_layers):
            is_last = (layer_i == num_layers - 1)
            per_head = (out_channels if is_last else hidden_channels) // heads
            conv_dict = {et: GATv2Conv(hidden_channels, per_head, heads=heads, concat=True, dropout=dropout, add_self_loops=False) for et in metadata[1]}
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(per_head * heads) for nt in metadata[0]}))

    def forward(self, x_dict, edge_index_dict):
        h = {nt: self.projections[nt](x) for nt, x in x_dict.items()}
        for i, (conv, norms) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                z = norms[nt](h_new[nt])
                if nt in h and h[nt].shape == z.shape: z = z + h[nt]
                if i < self.num_layers - 1: z = F.elu(z); z = F.dropout(z, p=self.dropout, training=self.training)
                h_new[nt] = z
            h = h_new
        return h

class HeteroAPPNPEncoder(nn.Module):
    """APPNP: Predict then Propagate (PPR ベース)"""
    def __init__(self, metadata, in_channels_dict, hidden_channels=128, out_channels=64, teleport_prob=0.15, num_iterations=10, dropout=0.2, **kwargs):
        super().__init__()
        self.alpha = teleport_prob; self.K = num_iterations; self.dropout = dropout
        self.edge_types = metadata[1]
        self.predict_mlps = nn.ModuleDict()
        for nt, in_dim in in_channels_dict.items():
            self.predict_mlps[nt] = nn.Sequential(
                nn.Linear(in_dim, hidden_channels), nn.LayerNorm(hidden_channels), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels),
            )

    def forward(self, x_dict, edge_index_dict):
        h0 = {nt: self.predict_mlps[nt](x) if nt in self.predict_mlps else x for nt, x in x_dict.items()}
        h = {nt: h0[nt].clone() for nt in h0}
        for _k in range(self.K):
            h_new = {nt: torch.zeros_like(h[nt]) for nt in h}
            nc = {nt: torch.zeros(h[nt].shape[0], 1, device=h[nt].device) for nt in h}
            for et in self.edge_types:
                st, _, dt = et; ei = edge_index_dict.get(et)
                if ei is None or ei.shape[1] == 0 or st not in h or dt not in h_new: continue
                h_new[dt].index_add_(0, ei[1], h[st][ei[0]])
                nc[dt].index_add_(0, ei[1], torch.ones(ei.shape[1], 1, device=ei.device))
            for nt in h_new:
                h[nt] = (1 - self.alpha) * (h_new[nt] / nc[nt].clamp(min=1)) + self.alpha * h0[nt]
        return h

class LinkPredictor(nn.Module):
    def __init__(self, embed_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))
    def forward(self, z_u, z_i): return self.net(torch.cat([z_u, z_i], dim=-1)).squeeze(-1)

class GNNRecommender(nn.Module):
    """encoder_type='gat' or 'appnp'"""
    def __init__(self, metadata, in_channels_dict, hidden_channels=128, out_channels=64, num_layers=3, heads=4, dropout=0.2, encoder_type="gat", teleport_prob=0.15, num_iterations=10):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "appnp":
            self.encoder = HeteroAPPNPEncoder(metadata, in_channels_dict, hidden_channels, out_channels, teleport_prob=teleport_prob, num_iterations=num_iterations, dropout=dropout)
        else:
            self.encoder = HeteroGATEncoder(metadata, in_channels_dict, hidden_channels, out_channels, num_layers, heads, dropout)
        self.predictor = LinkPredictor(out_channels, hidden_channels, dropout)

    def encode(self, x_dict, edge_index_dict): return self.encoder(x_dict, edge_index_dict)
    def predict_score(self, z_u, z_i): return self.predictor(z_u, z_i)

    def forward(self, x_dict, edge_index_dict, user_idx, pos_idx, neg_idx):
        z = self.encode(x_dict, edge_index_dict)
        pos_score = self.predict_score(z["user"][user_idx], z["item"][pos_idx])
        neg_score = self.predict_score(z["user"][user_idx], z["item"][neg_idx])
        return -F.logsigmoid(pos_score - neg_score).mean()

    @torch.no_grad()
    def get_embeddings(self, x_dict, edge_index_dict):
        z = self.encode(x_dict, edge_index_dict); return z["user"], z["item"]
