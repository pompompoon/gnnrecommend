#!/usr/bin/env python3
"""GNN レコメンド精度評価: Precision/Recall/NDCG/HitRate/MRR/MAP + Coverage/ILD/Novelty"""
from __future__ import annotations
import argparse, math, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from config import Config
from db.connection import DatabaseManager
from models.graph_builder import GraphBuilder
from models.gnn_model import GNNRecommender

class Evaluator:
    def __init__(self, model, builder, data, config):
        self.model = model; self.builder = builder; self.data = data
        self.db = DatabaseManager(config.db.connect_kwargs); self.device = torch.device(config.model.device)
        self.config = config; self.n_users = data["user"].num_nodes; self.n_items = data["item"].num_nodes
        self._split_edges(); self._precompute(); self._build_popularity(); self._build_item_category_map()

    def _split_edges(self):
        ei = self.data["user", "purchased", "item"].edge_index; n = ei.shape[1]; perm = torch.randperm(n)
        n_test = max(1, int(n * self.config.model.test_ratio))
        self.train_edges = ei[:, perm[:-n_test]]; self.test_edges = ei[:, perm[-n_test:]]
        self.train_pos: dict[int, set[int]] = {}
        for i in range(self.train_edges.shape[1]):
            u, it = self.train_edges[0,i].item(), self.train_edges[1,i].item()
            self.train_pos.setdefault(u, set()).add(it)
        if ("user", "viewed", "item") in self.data.edge_types:
            ve = self.data["user", "viewed", "item"].edge_index
            for i in range(ve.shape[1]): self.train_pos.setdefault(ve[0,i].item(), set()).add(ve[1,i].item())
        if ("user", "favorited", "item") in self.data.edge_types:
            fe = self.data["user", "favorited", "item"].edge_index
            for i in range(fe.shape[1]): self.train_pos.setdefault(fe[0,i].item(), set()).add(fe[1,i].item())
        self.test_gt: dict[int, set[int]] = {}
        for i in range(self.test_edges.shape[1]):
            u, it = self.test_edges[0,i].item(), self.test_edges[1,i].item()
            self.test_gt.setdefault(u, set()).add(it)
        total_pos = sum(len(v) for v in self.train_pos.values())
        print(f"  📊 購入エッジ分割: train={self.train_edges.shape[1]:,}  test={self.test_edges.shape[1]:,}")
        print(f"     全行動 positive set: {total_pos:,}  テスト対象ユーザー: {len(self.test_gt):,}")

    @torch.no_grad()
    def _precompute(self):
        self.model.eval(); self.model.to(self.device)
        x = {nt: self.data[nt].x.to(self.device) for nt in self.data.node_types}
        ei = {}
        for et in self.data.edge_types:
            idx = self.data[et].edge_index
            if et == ("user", "purchased", "item"): idx = self.train_edges
            ei[et] = idx.to(self.device)
        self.z_user, self.z_item = self.model.get_embeddings(x, ei)
        print(f"  ✅ Embedding 計算完了  user={self.z_user.shape}  item={self.z_item.shape}")

    def _build_popularity(self):
        counts = Counter()
        for i in range(self.train_edges.shape[1]): counts[self.train_edges[1,i].item()] += 1
        total = self.train_edges.shape[1]; min_p = 1.0 / (total + self.n_items)
        self.item_pop = {it: counts.get(it,0)/total if counts.get(it,0)>0 else min_p for it in range(self.n_items)}

    def _build_item_category_map(self):
        rows = self.db.fetch_all("SELECT i.item_id, i.subcategory_id, s.category_id FROM items i JOIN subcategories s ON i.subcategory_id = s.subcategory_id")
        iid_map = self.builder.item_id_map; self.item_subcat = {}; self.item_cat = {}
        for r in rows:
            g = iid_map.get(r["item_id"])
            if g is not None: self.item_subcat[g] = r["subcategory_id"]; self.item_cat[g] = r["category_id"]
        self.n_subcats = len(set(self.item_subcat.values())); self.n_cats = len(set(self.item_cat.values()))

    @torch.no_grad()
    def _get_topk(self, u, k):
        z_u = self.z_user[u].unsqueeze(0).expand(self.n_items, -1)
        scores = self.model.predict_score(z_u, self.z_item)
        for it in self.train_pos.get(u, set()): scores[it] = float("-inf")
        _, topk = torch.topk(scores, k); return topk.cpu().tolist()

    def compute_ranking_metrics(self, k_values):
        max_k = max(k_values)
        res = {k: {"precision":[], "recall":[], "ndcg":[], "hit":[], "mrr":[], "map":[]} for k in k_values}
        for u, true_items in self.test_gt.items():
            topk_list = self._get_topk(u, max_k)
            for k in k_values:
                topk = topk_list[:k]; topk_set = set(topk); n_hit = len(true_items & topk_set)
                res[k]["precision"].append(n_hit/k); res[k]["recall"].append(n_hit/len(true_items))
                res[k]["hit"].append(1.0 if n_hit>0 else 0.0)
                dcg = sum(1.0/math.log2(r+2) for r, item in enumerate(topk) if item in true_items)
                idcg = sum(1.0/math.log2(i+2) for i in range(min(len(true_items), k)))
                res[k]["ndcg"].append(dcg/max(idcg, 1e-10))
                rr = 0.0
                for r, item in enumerate(topk):
                    if item in true_items: rr = 1.0/(r+1); break
                res[k]["mrr"].append(rr)
                n_rel, sp = 0, 0.0
                for r, item in enumerate(topk):
                    if item in true_items: n_rel += 1; sp += n_rel/(r+1)
                res[k]["map"].append(sp/min(len(true_items), k))
        return {k: {n: float(np.mean(v)) for n, v in res[k].items()} for k in k_values}

    def compute_beyond_accuracy(self, k=10):
        all_items, all_cats, all_subcats, ilds, novs = set(), set(), set(), [], []
        for u in self.test_gt:
            topk = self._get_topk(u, k); all_items.update(topk)
            cats = [self.item_cat.get(it) for it in topk if self.item_cat.get(it) is not None]
            all_cats.update(cats); all_subcats.update(self.item_subcat.get(it) for it in topk if self.item_subcat.get(it))
            if len(cats) >= 2:
                np2, nd = 0, 0
                for i in range(len(cats)):
                    for j in range(i+1, len(cats)): np2 += 1; nd += (1 if cats[i] != cats[j] else 0)
                ilds.append(nd/np2)
            novs.append(np.mean([-math.log2(max(self.item_pop.get(it,1e-10),1e-10)) for it in topk]))
        return {"item_coverage": len(all_items)/self.n_items, "category_coverage": len(all_cats)/max(self.n_cats,1),
                "subcategory_coverage": len(all_subcats)/max(self.n_subcats,1), "avg_ILD": float(np.mean(ilds)) if ilds else 0,
                "avg_novelty": float(np.mean(novs)) if novs else 0, "unique_items": len(all_items), "total_items": self.n_items}

    def compute_by_user_activity(self, k=10):
        segs = {"light (1-5)":[], "medium (6-15)":[], "heavy (16+)":[]}
        for u, true_items in self.test_gt.items():
            n_train = len(self.train_pos.get(u, set())); topk = set(self._get_topk(u, k))
            hit = len(true_items & topk); recall = hit/len(true_items)
            dcg = sum(1.0/math.log2(r+2) for r, it in enumerate(self._get_topk(u,k)) if it in true_items)
            idcg = sum(1.0/math.log2(i+2) for i in range(min(len(true_items),k))); ndcg = dcg/max(idcg,1e-10)
            if n_train <= 5: segs["light (1-5)"].append((recall,ndcg))
            elif n_train <= 15: segs["medium (6-15)"].append((recall,ndcg))
            else: segs["heavy (16+)"].append((recall,ndcg))
        result = {}
        for seg, vals in segs.items():
            if vals: result[seg] = {"n_users": len(vals), f"recall@{k}": float(np.mean([v[0] for v in vals])), f"ndcg@{k}": float(np.mean([v[1] for v in vals]))}
            else: result[seg] = {"n_users": 0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}
        return result

    def compute_by_category(self, k=10):
        cat_names = {r["category_id"]: r["name"] for r in self.db.fetch_all("SELECT category_id, name FROM categories")}
        cat_hits: dict[int, list[float]] = {}
        for u, true_items in self.test_gt.items():
            topk_set = set(self._get_topk(u, k))
            for item in true_items:
                cat = self.item_cat.get(item)
                if cat is not None: cat_hits.setdefault(cat, []).append(1.0 if item in topk_set else 0.0)
        return {cat_names.get(cid, f"cat_{cid}"): {"n_test": len(h), f"hit_rate@{k}": float(np.mean(h))} for cid, h in cat_hits.items()}

    def full_report(self, k_values=None):
        if k_values is None: k_values = [5, 10, 20]
        L = []; sep = "=" * 72
        L += ["", sep, "  GNN レコメンドシステム 精度評価レポート", sep, ""]
        L += [f"📊 データ概要", f"  ユーザー: {self.n_users:,}  アイテム: {self.n_items:,}", f"  Train購入: {self.train_edges.shape[1]:,}  Test購入: {self.test_edges.shape[1]:,}  テスト対象: {len(self.test_gt):,}", ""]
        L += ["─"*72, "📈 ランキング指標", "─"*72]
        t0 = time.time(); ranking = self.compute_ranking_metrics(k_values); L += [f"  (計算時間: {time.time()-t0:.1f}s)", ""]
        L += [f"  {'K':>4s} │ {'Precision':>10s} │ {'Recall':>10s} │ {'NDCG':>10s} │ {'HitRate':>10s} │ {'MRR':>10s} │ {'MAP':>10s}"]
        L += ["  " + "─"*75]
        for k in k_values:
            m = ranking[k]; L += [f"  {k:>4d} │ {m['precision']:>10.4f} │ {m['recall']:>10.4f} │ {m['ndcg']:>10.4f} │ {m['hit']:>10.4f} │ {m['mrr']:>10.4f} │ {m['map']:>10.4f}"]
        bk = k_values[1] if len(k_values)>1 else k_values[0]; m = ranking[bk]
        L += ["", f"  💡 K={bk}: Recall={m['recall']:.4f} ({m['recall']*100:.1f}%捕捉)  HitRate={m['hit']:.4f} ({m['hit']*100:.1f}%的中)  MRR={m['mrr']:.4f} (平均{1/max(m['mrr'],1e-10):.0f}位)", ""]
        L += ["─"*72, "🌐 カバレッジ・多様性", "─"*72]
        ba = self.compute_beyond_accuracy(k=bk)
        L += [f"  Item Coverage: {ba['item_coverage']:.4f} ({ba['unique_items']:,}/{ba['total_items']:,})", f"  Category Coverage: {ba['category_coverage']:.4f}  SubCategory: {ba['subcategory_coverage']:.4f}", f"  Avg ILD (多様性): {ba['avg_ILD']:.4f}  Novelty: {ba['avg_novelty']:.2f} bits", ""]
        L += ["─"*72, f"👥 ユーザーセグメント別 (K={bk})", "─"*72]
        segs = self.compute_by_user_activity(k=bk)
        for seg, sm in segs.items(): L += [f"  {seg:>18s}  users={sm['n_users']:>5,}  R@{bk}={sm[f'recall@{bk}']:.4f}  N@{bk}={sm[f'ndcg@{bk}']:.4f}"]
        L += ["", "─"*72, f"🏷️  カテゴリ別 Hit Rate (K={bk})", "─"*72]
        cats = self.compute_by_category(k=bk)
        for cn, cm in sorted(cats.items(), key=lambda x: -x[1][f"hit_rate@{bk}"]): L += [f"  {cn:>20s}  n={cm['n_test']:>5,}  H@{bk}={cm[f'hit_rate@{bk}']:.4f}"]
        L += ["", sep]; return "\n".join(L)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=str, default="5,10,20"); p.add_argument("--output", type=str, default=None)
    p.add_argument("--encoder", choices=["gat","appnp"], default="gat"); p.add_argument("--teleport", type=float, default=0.15); p.add_argument("--ppr-iters", type=int, default=10)
    p.add_argument("--db-host", default="localhost"); p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--db-name", default="fashion_recommend"); p.add_argument("--db-user", default="postgres"); p.add_argument("--db-password", default="postgres")
    args = p.parse_args()
    cfg = Config(); cfg.db.host=args.db_host; cfg.db.port=args.db_port; cfg.db.dbname=args.db_name; cfg.db.user=args.db_user; cfg.db.password=args.db_password
    cfg.model.encoder_type=args.encoder; cfg.model.teleport_prob=args.teleport; cfg.model.num_iterations=args.ppr_iters
    if torch.cuda.is_available(): cfg.model.device = "cuda"
    k_values = [int(x.strip()) for x in args.top_k.split(",")]
    print("📐 グラフ構築 & モデルロード中...")
    builder = GraphBuilder(cfg.db.connect_kwargs); data = builder.build()
    in_ch = {nt: data[nt].x.shape[1] for nt in data.node_types}; mc = cfg.model
    model = GNNRecommender(metadata=data.metadata(), in_channels_dict=in_ch, hidden_channels=mc.hidden_channels, out_channels=mc.out_channels, num_layers=mc.num_layers, heads=mc.heads, dropout=mc.dropout, encoder_type=mc.encoder_type, teleport_prob=mc.teleport_prob, num_iterations=mc.num_iterations)
    ckpt = Path("checkpoints/best_model.pt")
    if ckpt.exists(): model.load_state_dict(torch.load(ckpt, map_location=mc.device, weights_only=True)); print(f"  ✅ チェックポイント読込: {ckpt}")
    else: print("  ⚠️  チェックポイント未検出")
    print("\n🔬 評価開始..."); evaluator = Evaluator(model, builder, data, cfg); report = evaluator.full_report(k_values); print(report)
    if args.output: Path(args.output).write_text(report, encoding="utf-8"); print(f"\n📄 レポート保存: {args.output}")

if __name__ == "__main__": main()
