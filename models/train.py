"""
学習パイプライン

  - 購入エッジを Train / Val / Test に分割
  - BPR ネガティブサンプリング
  - Recall@K / NDCG@K / HitRate@K で評価
  - Early stopping + cosine annealing LR
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData

from config import ModelConfig
from models.gnn_model import GNNRecommender


class Trainer:
    """モデル学習の管理"""

    def __init__(
        self,
        model: GNNRecommender,
        data: HeteroData,
        config: ModelConfig,
    ):
        self.model = model
        self.data = data
        self.cfg = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6,
        )

        self._split_edges()

    # --------------------------------------------------
    # Edge splitting
    # --------------------------------------------------

    def _split_edges(self) -> None:
        """購入エッジを train / val / test に分割"""
        ei = self.data["user", "purchased", "item"].edge_index
        n = ei.shape[1]
        perm = torch.randperm(n)

        n_val = int(n * self.cfg.val_ratio)
        n_test = int(n * self.cfg.test_ratio)

        self.train_edges = ei[:, perm[: n - n_val - n_test]]
        self.val_edges = ei[:, perm[n - n_val - n_test : n - n_test]]
        self.test_edges = ei[:, perm[n - n_test :]]

        self.n_items = self.data["item"].num_nodes

        # positive set (高速 lookup)
        self.pos_set: set[tuple[int, int]] = set()
        for i in range(self.train_edges.shape[1]):
            u = self.train_edges[0, i].item()
            it = self.train_edges[1, i].item()
            self.pos_set.add((u, it))

        n_train = self.train_edges.shape[1]
        print(f"  📊 エッジ分割: train={n_train:,}  val={n_val:,}  test={n_test:,}")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _negative_sample(self, users: torch.Tensor) -> torch.Tensor:
        negs = torch.randint(0, self.n_items, users.shape)
        for i, u in enumerate(users.tolist()):
            while (u, negs[i].item()) in self.pos_set:
                negs[i] = torch.randint(0, self.n_items, (1,))
        return negs

    def _to_device(self) -> tuple[dict, dict]:
        x = {nt: self.data[nt].x.to(self.device) for nt in self.data.node_types}
        ei = {}
        for et in self.data.edge_types:
            idx = self.data[et].edge_index
            # 購入エッジは train 分のみ
            if et == ("user", "purchased", "item"):
                idx = self.train_edges
            ei[et] = idx.to(self.device)
        return x, ei

    # --------------------------------------------------
    # Train / Eval
    # --------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        x, ei = self._to_device()

        users = self.train_edges[0]
        pos = self.train_edges[1]
        neg = self._negative_sample(users)

        loader = DataLoader(
            TensorDataset(users, pos, neg),
            batch_size=self.cfg.batch_size, shuffle=True,
        )

        total_loss, n_batch = 0.0, 0
        for bu, bp, bn in loader:
            bu, bp, bn = bu.to(self.device), bp.to(self.device), bn.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(x, ei, bu, bp, bn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        self.scheduler.step()
        return total_loss / max(n_batch, 1)

    @torch.no_grad()
    def evaluate(self, edges: torch.Tensor, k: int = 10) -> dict[str, float]:
        self.model.eval()
        x, ei = self._to_device()
        z_user, z_item = self.model.get_embeddings(x, ei)

        # ユーザーごとの正解アイテム
        ground_truth: dict[int, set[int]] = {}
        for i in range(edges.shape[1]):
            u, it = edges[0, i].item(), edges[1, i].item()
            ground_truth.setdefault(u, set()).add(it)

        recalls, ndcgs, hits = [], [], []
        for u, true_items in ground_truth.items():
            z_u = z_user[u].unsqueeze(0).expand(z_item.shape[0], -1)
            scores = self.model.predict_score(z_u, z_item)
            # train で見たものを除外
            for it in range(self.n_items):
                if (u, it) in self.pos_set:
                    scores[it] = float("-inf")

            _, topk = torch.topk(scores, k)
            topk_set = set(topk.cpu().tolist())

            hit = len(true_items & topk_set)
            recalls.append(hit / min(len(true_items), k))
            hits.append(1.0 if hit > 0 else 0.0)

            dcg = sum(1.0 / np.log2(r + 2) for r, idx in enumerate(topk.cpu().tolist()) if idx in true_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcgs.append(dcg / max(idcg, 1e-10))

        return {
            f"recall@{k}": float(np.mean(recalls)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
            f"hit@{k}": float(np.mean(hits)),
        }

    # --------------------------------------------------
    # Full training loop
    # --------------------------------------------------

    def train(self, epochs: int | None = None, save_dir: str = "checkpoints") -> dict:
        epochs = epochs or self.cfg.epochs
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        best_ndcg = 0.0
        best_metrics: dict = {}
        patience_cnt = 0

        print(f"\n🚀 学習開始  epochs={epochs}  device={self.device}")
        print("─" * 72)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = self._train_epoch()
            dt = time.time() - t0

            if epoch % 5 == 0 or epoch == 1:
                val = self.evaluate(self.val_edges, k=10)
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:4d} │ loss={loss:.4f} │ "
                    f"R@10={val['recall@10']:.4f}  N@10={val['ndcg@10']:.4f}  "
                    f"H@10={val['hit@10']:.4f} │ lr={lr:.2e} │ {dt:.1f}s"
                )
                if val["ndcg@10"] > best_ndcg:
                    best_ndcg = val["ndcg@10"]
                    best_metrics = val
                    patience_cnt = 0
                    torch.save(self.model.state_dict(), save_path / "best_model.pt")
                    print(f"         ✨ best model saved (ndcg={best_ndcg:.4f})")
                else:
                    patience_cnt += 5
                    if patience_cnt >= self.cfg.early_stop_patience:
                        print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                        break

        # Test
        print("─" * 72)
        ckpt = save_path / "best_model.pt"
        if ckpt.exists():
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        test = self.evaluate(self.test_edges, k=10)
        print(f"  📈 テスト結果:  Recall@10={test['recall@10']:.4f}  "
              f"NDCG@10={test['ndcg@10']:.4f}  Hit@10={test['hit@10']:.4f}")
        return test
