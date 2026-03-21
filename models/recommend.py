"""レコメンド推論エンジン"""
from __future__ import annotations
import torch
from torch_geometric.data import HeteroData
from config import Config
from db.connection import DatabaseManager
from models.graph_builder import GraphBuilder
from models.gnn_model import GNNRecommender

class Recommender:
    def __init__(self, model: GNNRecommender, builder: GraphBuilder, data: HeteroData, config: Config):
        self.model = model; self.builder = builder; self.data = data
        self.db = DatabaseManager(config.db.connect_kwargs); self.device = torch.device(config.model.device)
        self._precompute()

    @torch.no_grad()
    def _precompute(self):
        self.model.eval()
        x = {nt: self.data[nt].x.to(self.device) for nt in self.data.node_types}
        ei = {et: self.data[et].edge_index.to(self.device) for et in self.data.edge_types}
        self.z_user, self.z_item = self.model.get_embeddings(x, ei)
        print(f"  ✅ Embedding 事前計算完了  user={self.z_user.shape}  item={self.z_item.shape}")

    @torch.no_grad()
    def recommend_for_user(self, user_id, top_k=10, exclude_purchased=True):
        uid_map = self.builder.user_id_map; iid_rev = self.builder.reverse_item_map
        if user_id not in uid_map: print(f"  ⚠️  user_id={user_id} not found"); return []
        g_uid = uid_map[user_id]
        z_u = self.z_user[g_uid].unsqueeze(0).expand(self.z_item.shape[0], -1)
        scores = self.model.predict_score(z_u, self.z_item)
        if exclude_purchased:
            rows = self.db.fetch_all("SELECT DISTINCT item_id FROM purchases WHERE user_id=%s", (user_id,))
            for r in rows:
                g_iid = self.builder.item_id_map.get(r["item_id"])
                if g_iid is not None: scores[g_iid] = float("-inf")
        top_scores, top_idx = torch.topk(scores, top_k)
        results = []
        for sv, idx in zip(top_scores.cpu().tolist(), top_idx.cpu().tolist()):
            db_iid = iid_rev.get(idx)
            if db_iid is None: continue
            info = self.db.fetch_one("SELECT i.item_id, i.name, i.price, i.color, i.season, i.is_on_sale, s.name AS subcategory, c.name AS category, b.name AS brand FROM items i JOIN subcategories s ON i.subcategory_id = s.subcategory_id JOIN categories c ON s.category_id = c.category_id LEFT JOIN brands b ON i.brand_id = b.brand_id WHERE i.item_id = %s", (db_iid,))
            if info: results.append({"rank": len(results)+1, "item_id": info["item_id"], "name": info["name"], "category": info["category"], "subcategory": info["subcategory"], "brand": info["brand"], "price": int(info["price"]), "color": info["color"], "season": info["season"], "is_on_sale": info["is_on_sale"], "score": round(sv, 5)})
        return results

    @torch.no_grad()
    def find_similar_items(self, item_id, top_k=10):
        iid_map = self.builder.item_id_map; iid_rev = self.builder.reverse_item_map
        if item_id not in iid_map: return []
        g_iid = iid_map[item_id]; z_t = self.z_item[g_iid]
        norms = self.z_item.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims = ((self.z_item / norms) @ (z_t / z_t.norm().clamp(min=1e-8)).unsqueeze(1)).squeeze(1)
        sims[g_iid] = -1.0
        top_sim, top_idx = torch.topk(sims, top_k)
        results = []
        for sv, idx in zip(top_sim.cpu().tolist(), top_idx.cpu().tolist()):
            db_iid = iid_rev.get(idx)
            if db_iid is None: continue
            info = self.db.fetch_one("SELECT i.item_id, i.name, i.price, i.color, s.name AS subcategory, c.name AS category, b.name AS brand FROM items i JOIN subcategories s ON i.subcategory_id = s.subcategory_id JOIN categories c ON s.category_id = c.category_id LEFT JOIN brands b ON i.brand_id = b.brand_id WHERE i.item_id = %s", (db_iid,))
            if info: results.append({"rank": len(results)+1, "item_id": info["item_id"], "name": info["name"], "category": info["category"], "subcategory": info["subcategory"], "brand": info["brand"], "price": int(info["price"]), "color": info["color"], "similarity": round(sv, 5)})
        return results

    def save_to_db(self, user_ids=None, top_k=20, model_version="v1.0"):
        if user_ids is None: user_ids = [r["user_id"] for r in self.db.fetch_all("SELECT user_id FROM users ORDER BY user_id")]
        self.db.execute("DELETE FROM recommendations WHERE model_version=%s", (model_version,))
        total = 0
        for uid in user_ids:
            recs = self.recommend_for_user(uid, top_k)
            if recs:
                vals = [(uid, r["item_id"], r["score"], r["rank"], model_version) for r in recs]
                self.db.execute_values("INSERT INTO recommendations (user_id, item_id, score, rank, model_version) VALUES %s", vals)
                total += len(vals)
        print(f"  ✅ {len(user_ids):,} ユーザー × top-{top_k} → {total:,} 件を DB に保存")

    def print_user_recommendations(self, user_id, top_k=10):
        u = self.db.fetch_one("SELECT username, age, gender, prefecture FROM users WHERE user_id=%s", (user_id,))
        if u: print(f"\n  👤 {u['username']}  (ID={user_id}, {u['age']}歳, {u['gender']}, {u['prefecture']})")
        history = self.db.fetch_all("SELECT i.name, s.name AS sub, b.name AS brand, i.price FROM purchases p JOIN items i ON p.item_id = i.item_id JOIN subcategories s ON i.subcategory_id = s.subcategory_id LEFT JOIN brands b ON i.brand_id = b.brand_id WHERE p.user_id = %s ORDER BY p.purchased_at DESC LIMIT 5", (user_id,))
        if history:
            print("  📦 最近の購入:")
            for h in history: print(f"     ・{h['name']}  ({h['sub']})  ¥{h['price']:,}")
        recs = self.recommend_for_user(user_id, top_k)
        if recs:
            print(f"\n  🎯 レコメンド Top-{top_k}:")
            for r in recs:
                sale = " 🔥SALE" if r.get("is_on_sale") else ""
                print(f"     {r['rank']:2d}. {r['name']}  [{r['category']}>{r['subcategory']}]  ¥{r['price']:,}  score={r['score']:.4f}{sale}")
        else: print("  レコメンドなし")
        print()
