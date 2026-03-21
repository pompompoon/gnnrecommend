"""学習パイプライン (Multi-Behavior): 購入+閲覧+お気に入りを重み付きBPRで学習"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData
from config import ModelConfig
from models.gnn_model import GNNRecommender

BEHAVIOR_WEIGHTS = {"purchased": 3.0, "favorited": 2.0, "viewed": 1.0}

class Trainer:
    def __init__(self, model: GNNRecommender, data: HeteroData, config: ModelConfig):
        self.model = model; self.data = data; self.cfg = config
        self.device = torch.device(config.device); self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)
        self.n_items = data["item"].num_nodes
        self._split_edges(); self._build_training_data()

    def _split_edges(self):
        ei = self.data["user", "purchased", "item"].edge_index; n = ei.shape[1]
        perm = torch.randperm(n); n_val = int(n * self.cfg.val_ratio); n_test = int(n * self.cfg.test_ratio)
        self.purchase_train = ei[:, perm[:n - n_val - n_test]]
        self.val_edges = ei[:, perm[n - n_val - n_test:n - n_test]]
        self.test_edges = ei[:, perm[n - n_test:]]
        print(f"  📊 購入エッジ分割: train={self.purchase_train.shape[1]:,}  val={self.val_edges.shape[1]:,}  test={self.test_edges.shape[1]:,}")

    def _build_training_data(self):
        all_u, all_i, all_w = [], [], []
        n_p = self.purchase_train.shape[1]
        all_u.append(self.purchase_train[0]); all_i.append(self.purchase_train[1]); all_w.append(torch.full((n_p,), BEHAVIOR_WEIGHTS["purchased"]))
        if ("user", "viewed", "item") in self.data.edge_types:
            ve = self.data["user", "viewed", "item"].edge_index; n_v = ve.shape[1]
            all_u.append(ve[0]); all_i.append(ve[1]); all_w.append(torch.full((n_v,), BEHAVIOR_WEIGHTS["viewed"]))
            print(f"  📖 閲覧エッジを学習ターゲットに追加: {n_v:,}")
        if ("user", "favorited", "item") in self.data.edge_types:
            fe = self.data["user", "favorited", "item"].edge_index; n_f = fe.shape[1]
            all_u.append(fe[0]); all_i.append(fe[1]); all_w.append(torch.full((n_f,), BEHAVIOR_WEIGHTS["favorited"]))
            print(f"  ❤️  お気に入りエッジを学習ターゲットに追加: {n_f:,}")
        self.train_users = torch.cat(all_u); self.train_items = torch.cat(all_i); self.train_weights = torch.cat(all_w)
        self.pos_set: set[tuple[int, int]] = set()
        for i in range(len(self.train_users)): self.pos_set.add((self.train_users[i].item(), self.train_items[i].item()))
        total = len(self.train_users); density = total / (self.data["user"].num_nodes * self.n_items) * 100
        print(f"  🔗 統合学習エッジ数: {total:,}  (密度: {density:.1f}%)")

    def _negative_sample(self, users):
        negs = torch.randint(0, self.n_items, users.shape)
        for i, u in enumerate(users.tolist()):
            att = 0
            while (u, negs[i].item()) in self.pos_set and att < 20: negs[i] = torch.randint(0, self.n_items, (1,)); att += 1
        return negs

    def _to_device(self):
        x = {nt: self.data[nt].x.to(self.device) for nt in self.data.node_types}
        ei = {}
        for et in self.data.edge_types:
            idx = self.data[et].edge_index
            if et == ("user", "purchased", "item"): idx = self.purchase_train
            ei[et] = idx.to(self.device)
        return x, ei

    def _train_epoch(self):
        self.model.train(); x, ei = self._to_device()
        perm = torch.randperm(len(self.train_users))
        users, pos_items, weights = self.train_users[perm], self.train_items[perm], self.train_weights[perm]
        neg_items = self._negative_sample(users)
        loader = DataLoader(TensorDataset(users, pos_items, neg_items, weights), batch_size=self.cfg.batch_size, shuffle=False)
        total_loss, n_batch = 0.0, 0
        for bu, bp, bn, bw in loader:
            bu, bp, bn, bw = bu.to(self.device), bp.to(self.device), bn.to(self.device), bw.to(self.device)
            self.optimizer.zero_grad()
            z = self.model.encode(x, ei)
            pos_score = self.model.predict_score(z["user"][bu], z["item"][bp])
            neg_score = self.model.predict_score(z["user"][bu], z["item"][bn])
            loss = (-F.logsigmoid(pos_score - neg_score) * bw).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
            total_loss += loss.item(); n_batch += 1
        self.scheduler.step(); return total_loss / max(n_batch, 1)

    @torch.no_grad()
    def evaluate(self, edges, k=10):
        self.model.eval(); x, ei = self._to_device()
        z_user, z_item = self.model.get_embeddings(x, ei)
        gt: dict[int, set[int]] = {}
        for i in range(edges.shape[1]): u, it = edges[0,i].item(), edges[1,i].item(); gt.setdefault(u, set()).add(it)
        recalls, ndcgs, hits = [], [], []
        for u, true_items in gt.items():
            z_u = z_user[u].unsqueeze(0).expand(z_item.shape[0], -1)
            scores = self.model.predict_score(z_u, z_item)
            for it in range(self.n_items):
                if (u, it) in self.pos_set: scores[it] = float("-inf")
            _, topk = torch.topk(scores, k); topk_set = set(topk.cpu().tolist())
            hit = len(true_items & topk_set); recalls.append(hit / min(len(true_items), k)); hits.append(1.0 if hit > 0 else 0.0)
            dcg = sum(1.0/np.log2(r+2) for r, idx in enumerate(topk.cpu().tolist()) if idx in true_items)
            idcg = sum(1.0/np.log2(i+2) for i in range(min(len(true_items), k))); ndcgs.append(dcg/max(idcg, 1e-10))
        return {f"recall@{k}": float(np.mean(recalls)), f"ndcg@{k}": float(np.mean(ndcgs)), f"hit@{k}": float(np.mean(hits))}

    def train(self, epochs=None, save_dir="checkpoints"):
        epochs = epochs or self.cfg.epochs; save_path = Path(save_dir); save_path.mkdir(exist_ok=True)
        best_ndcg, patience_cnt = 0.0, 0
        print(f"\n🚀 学習開始  epochs={epochs}  device={self.device}")
        print(f"   行動重み: purchased={BEHAVIOR_WEIGHTS['purchased']:.1f}  favorited={BEHAVIOR_WEIGHTS['favorited']:.1f}  viewed={BEHAVIOR_WEIGHTS['viewed']:.1f}")
        print("─" * 72)
        for epoch in range(1, epochs + 1):
            t0 = time.time(); loss = self._train_epoch(); dt = time.time() - t0
            if epoch % 5 == 0 or epoch == 1:
                val = self.evaluate(self.val_edges, k=10); lr = self.scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:4d} │ loss={loss:.4f} │ R@10={val['recall@10']:.4f}  N@10={val['ndcg@10']:.4f}  H@10={val['hit@10']:.4f} │ lr={lr:.2e} │ {dt:.1f}s")
                if val["ndcg@10"] > best_ndcg:
                    best_ndcg = val["ndcg@10"]; patience_cnt = 0
                    torch.save(self.model.state_dict(), save_path / "best_model.pt"); print(f"         ✨ best model saved (ndcg={best_ndcg:.4f})")
                else:
                    patience_cnt += 5
                    if patience_cnt >= self.cfg.early_stop_patience: print(f"\n  ⏹️  Early stopping at epoch {epoch}"); break
        print("─" * 72)
        ckpt = save_path / "best_model.pt"
        if ckpt.exists(): self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        test = self.evaluate(self.test_edges, k=10)
        print(f"  📈 テスト結果:  Recall@10={test['recall@10']:.4f}  NDCG@10={test['ndcg@10']:.4f}  Hit@10={test['hit@10']:.4f}")
        return test
