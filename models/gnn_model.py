"""
GNN レコメンドモデル

  HeteroGATEncoder
    - 各ノードタイプの特徴量を hidden_channels に射影
    - GATv2Conv × num_layers でメッセージパッシング
    - Residual + LayerNorm

  LinkPredictor
    - user ⊕ item → MLP → score

  GNNRecommender
    - BPR Loss で学習
    - user / item embedding の内積で推論
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


# =================================================================
# Encoder
# =================================================================

class HeteroGATEncoder(nn.Module):
    """
    ヘテロジニアス GATv2 エンコーダー

    Parameters
    ----------
    metadata        : (node_types, edge_types) from HeteroData.metadata()
    in_channels_dict: {node_type: input_dim}
    hidden_channels : int
    out_channels    : int  ← 最終出力の次元
    num_layers      : int
    heads           : int  ← GATv2 の attention heads 数
    dropout         : float
    """

    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 入力射影: 各ノードタイプ → hidden_channels
        self.projections = nn.ModuleDict({
            nt: Linear(dim, hidden_channels)
            for nt, dim in in_channels_dict.items()
        })

        # GATv2Conv × num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_i in range(num_layers):
            is_last = (layer_i == num_layers - 1)
            per_head = (out_channels if is_last else hidden_channels) // heads

            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            norm_dim = per_head * heads
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(norm_dim) for nt in metadata[0]
            }))

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        # 入力射影
        h = {nt: self.projections[nt](x) for nt, x in x_dict.items()}

        # GATv2 layers
        for i, (conv, norms) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index_dict)

            for nt in h_new:
                z = norms[nt](h_new[nt])
                # residual（同一次元なら）
                if nt in h and h[nt].shape == z.shape:
                    z = z + h[nt]
                if i < self.num_layers - 1:
                    z = F.elu(z)
                    z = F.dropout(z, p=self.dropout, training=self.training)
                h_new[nt] = z
            h = h_new

        return h


# =================================================================
# Link Predictor (MLP)
# =================================================================

class LinkPredictor(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_u: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_u, z_i], dim=-1)).squeeze(-1)


# =================================================================
# Full Model
# =================================================================

class GNNRecommender(nn.Module):
    """
    GNN レコメンドモデル (Encoder + LinkPredictor)

    学習: BPR Loss (Bayesian Personalized Ranking)
    推論: user embedding × item embedding → score → top-k
    """

    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = HeteroGATEncoder(
            metadata, in_channels_dict,
            hidden_channels, out_channels, num_layers, heads, dropout,
        )
        self.predictor = LinkPredictor(out_channels, hidden_channels, dropout)

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def predict_score(self, z_u: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        return self.predictor(z_u, z_i)

    def forward(self, x_dict, edge_index_dict, user_idx, pos_idx, neg_idx):
        """BPR Loss の計算"""
        z = self.encode(x_dict, edge_index_dict)
        z_u = z["user"][user_idx]
        z_pos = z["item"][pos_idx]
        z_neg = z["item"][neg_idx]

        pos_score = self.predict_score(z_u, z_pos)
        neg_score = self.predict_score(z_u, z_neg)

        return -F.logsigmoid(pos_score - neg_score).mean()

    @torch.no_grad()
    def get_embeddings(self, x_dict, edge_index_dict):
        """全ノードの embedding を取得"""
        z = self.encode(x_dict, edge_index_dict)
        return z["user"], z["item"]
