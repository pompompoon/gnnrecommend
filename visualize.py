#!/usr/bin/env python3
"""GNN レコメンド結果の可視化: PostgreSQL → NetworkX → pyvis/matplotlib"""
from __future__ import annotations
import argparse
from pathlib import Path
import networkx as nx
from config import Config
from db.connection import DatabaseManager

STYLE = {
    "user":        {"color": "#f59e0b", "size": 40, "shape": "dot",     "font_color": "#f59e0b"},
    "item":        {"color": "#3b82f6", "size": 20, "shape": "dot",     "font_color": "#93c5fd"},
    "item_rec":    {"color": "#10b981", "size": 25, "shape": "star",    "font_color": "#6ee7b7"},
    "subcategory": {"color": "#8b5cf6", "size": 18, "shape": "diamond", "font_color": "#c4b5fd"},
    "category":    {"color": "#ec4899", "size": 30, "shape": "triangle","font_color": "#f9a8d4"},
}
EDGE_STYLE = {
    "purchased":  {"color": "#f59e0b", "width": 2.5, "dashes": False},
    "viewed":     {"color": "#64748b", "width": 1.0, "dashes": [5, 5]},
    "favorited":  {"color": "#f472b6", "width": 1.5, "dashes": [2, 2]},
    "recommend":  {"color": "#10b981", "width": 3.0, "dashes": [8, 4]},
    "belongs_to": {"color": "#475569", "width": 0.8, "dashes": False},
    "child_of":   {"color": "#475569", "width": 0.8, "dashes": False},
}

def _short(name, n=18): return name[:n]+"…" if len(name)>n else name

class RecommendationGraphBuilder:
    def __init__(self, config): self.db = DatabaseManager(config.db.connect_kwargs)

    def build_user_graph(self, user_id, recommendations, show_views=False, show_favorites=True, show_categories=True):
        G = nx.DiGraph()
        user = self.db.fetch_one("SELECT user_id, username, age, gender, prefecture FROM users WHERE user_id=%s", (user_id,))
        if not user: return G
        uid = f"user_{user_id}"
        G.add_node(uid, label=user["username"], title=f"👤 {user['username']}\n{user['age']}歳/{user['gender']}/{user['prefecture']}", node_type="user", **STYLE["user"])
        purchases = self.db.fetch_all("SELECT DISTINCT i.item_id, i.name, i.price, i.color, s.name AS subcategory, s.subcategory_id, c.name AS category, c.category_id FROM purchases p JOIN items i ON p.item_id=i.item_id JOIN subcategories s ON i.subcategory_id=s.subcategory_id JOIN categories c ON s.category_id=c.category_id WHERE p.user_id=%s", (user_id,))
        pid = set()
        for it in purchases:
            nid = f"item_{it['item_id']}"; pid.add(it["item_id"])
            G.add_node(nid, label=_short(it["name"]), title=f"📦 {it['name']}\n¥{it['price']:,}/{it['color']}", node_type="item", **STYLE["item"])
            G.add_edge(uid, nid, label="購入", edge_type="purchased", **EDGE_STYLE["purchased"])
            if show_categories: self._add_cats(G, it)
        for rec in recommendations:
            nid = f"item_{rec['item_id']}"
            if rec["item_id"] in pid: continue
            s = f"{rec['score']:.3f}" if isinstance(rec['score'], float) else str(rec['score'])
            G.add_node(nid, label=f"#{rec['rank']} {_short(rec['name'])}", title=f"🎯 #{rec['rank']}\n{rec['name']}\n¥{rec['price']:,}\nscore:{s}", node_type="item_rec", **STYLE["item_rec"])
            G.add_edge(uid, nid, label=f"rec#{rec['rank']}", edge_type="recommend", **EDGE_STYLE["recommend"])
            if show_categories:
                info = self.db.fetch_one("SELECT s.name AS subcategory, s.subcategory_id, c.name AS category, c.category_id FROM items i JOIN subcategories s ON i.subcategory_id=s.subcategory_id JOIN categories c ON s.category_id=c.category_id WHERE i.item_id=%s", (rec["item_id"],))
                if info: self._add_cats(G, {**rec, **info})
        return G

    def _add_cats(self, G, info):
        inid = f"item_{info['item_id']}"; snid = f"sub_{info['subcategory_id']}"; cnid = f"cat_{info['category_id']}"
        if not G.has_node(snid): G.add_node(snid, label=info["subcategory"], title=f"📂 {info['subcategory']}", node_type="subcategory", **STYLE["subcategory"])
        if not G.has_node(cnid): G.add_node(cnid, label=info["category"], title=f"🏷️ {info['category']}", node_type="category", **STYLE["category"])
        if not G.has_edge(inid, snid): G.add_edge(inid, snid, edge_type="belongs_to", **EDGE_STYLE["belongs_to"])
        if not G.has_edge(snid, cnid): G.add_edge(snid, cnid, edge_type="child_of", **EDGE_STYLE["child_of"])

def visualize_pyvis(G, output="network.html", title="GNN Recommendation Network"):
    from pyvis.network import Network
    net = Network(height="900px", width="100%", directed=True, bgcolor="#0a0e17", font_color="#e2e8f0")
    net.set_options('{"physics":{"forceAtlas2Based":{"gravitationalConstant":-120,"springLength":160},"solver":"forceAtlas2Based","stabilization":{"iterations":200}},"edges":{"arrows":{"to":{"enabled":true,"scaleFactor":0.5}},"smooth":{"type":"curvedCW","roundness":0.15}},"interaction":{"hover":true,"navigationButtons":true}}')
    for nid, a in G.nodes(data=True): net.add_node(nid, label=a.get("label",""), title=a.get("title",""), color=a.get("color","#888"), size=a.get("size",20), shape=a.get("shape","dot"), font={"color": a.get("font_color","#e2e8f0")})
    for s, d, a in G.edges(data=True): net.add_edge(s, d, title=a.get("title",""), label=a.get("label",""), color=a.get("color","#475569"), width=a.get("width",1), dashes=a.get("dashes",False))
    net.save_graph(output); print(f"  ✅ pyvis HTML: {output}"); return output

def visualize_matplotlib(G, output="network.png", title="GNN Recommendation Network"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try: import japanize_matplotlib
    except: plt.rcParams["font.family"] = "Noto Sans JP"
    fig, ax = plt.subplots(figsize=(18,12), facecolor="#0a0e17"); ax.set_facecolor("#0a0e17")
    layer_x = {"user": 0, "item": 1.5, "item_rec": 1.5, "subcategory": 3, "category": 4.5}
    nbt = {}
    for n, d in G.nodes(data=True): nbt.setdefault(d.get("node_type","item"), []).append(n)
    pos = {}
    for nt, nodes in nbt.items():
        x = layer_x.get(nt, 1.5)
        for i, node in enumerate(sorted(nodes)): pos[node] = (x, (i - len(nodes)/2)*0.8)
    for nt, nodes in nbt.items():
        s = STYLE.get(nt, {"color":"#888","size":20}); colors = [G.nodes[n].get("color", s["color"]) for n in nodes]
        sizes = [G.nodes[n].get("size", s["size"])*15 for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=sizes, alpha=0.9, ax=ax)
    for etype in EDGE_STYLE:
        edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("edge_type")==etype]
        if edges:
            s = EDGE_STYLE[etype]; ls = "dashed" if etype in ("recommend","viewed","favorited") else "solid"
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=s["color"], width=s["width"], alpha=0.6, style=ls, arrows=True, arrowsize=12, ax=ax, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(G, pos, {n: d.get("label","") for n,d in G.nodes(data=True)}, font_size=8, font_color="#e2e8f0", ax=ax)
    ax.set_title(title, fontsize=16, color="#10b981", pad=15); ax.axis("off"); plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor="#0a0e17", bbox_inches="tight"); plt.close()
    print(f"  ✅ matplotlib: {output}"); return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", type=int, default=None); p.add_argument("--user-ids", type=str, default=None)
    p.add_argument("--item-id", type=int, default=None); p.add_argument("--similar", action="store_true")
    p.add_argument("--top-k", type=int, default=5); p.add_argument("--output", type=str, default="network.html")
    p.add_argument("--static", action="store_true"); p.add_argument("--show-views", action="store_true"); p.add_argument("--no-categories", action="store_true")
    p.add_argument("--encoder", choices=["gat","appnp"], default="gat"); p.add_argument("--teleport", type=float, default=0.15); p.add_argument("--ppr-iters", type=int, default=10)
    p.add_argument("--db-host", default="localhost"); p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--db-name", default="fashion_recommend"); p.add_argument("--db-user", default="postgres"); p.add_argument("--db-password", default="postgres")
    args = p.parse_args()
    cfg = Config(); cfg.db.host=args.db_host; cfg.db.port=args.db_port; cfg.db.dbname=args.db_name; cfg.db.user=args.db_user; cfg.db.password=args.db_password
    cfg.model.encoder_type=args.encoder; cfg.model.teleport_prob=args.teleport; cfg.model.num_iterations=args.ppr_iters
    import torch
    from models.graph_builder import GraphBuilder
    from models.gnn_model import GNNRecommender
    from models.recommend import Recommender
    print("📐 グラフ構築 & モデルロード中...")
    builder = GraphBuilder(cfg.db.connect_kwargs); data = builder.build()
    in_ch = {nt: data[nt].x.shape[1] for nt in data.node_types}; mc = cfg.model
    model = GNNRecommender(metadata=data.metadata(), in_channels_dict=in_ch, hidden_channels=mc.hidden_channels, out_channels=mc.out_channels, num_layers=mc.num_layers, heads=mc.heads, dropout=mc.dropout, encoder_type=mc.encoder_type, teleport_prob=mc.teleport_prob, num_iterations=mc.num_iterations)
    ckpt = Path("checkpoints/best_model.pt")
    if ckpt.exists(): model.load_state_dict(torch.load(ckpt, map_location=mc.device, weights_only=True)); print(f"  ✅ チェックポイント読込: {ckpt}")
    rec = Recommender(model, builder, data, cfg); gb = RecommendationGraphBuilder(cfg)
    uid = args.user_id
    if uid is None:
        row = gb.db.fetch_one("SELECT user_id FROM users ORDER BY user_id LIMIT 1")
        uid = row["user_id"] if row else 1
    recs = rec.recommend_for_user(uid, args.top_k)
    G = gb.build_user_graph(uid, recs, show_views=args.show_views, show_categories=not args.no_categories)
    uinfo = gb.db.fetch_one("SELECT username FROM users WHERE user_id=%s", (uid,))
    title = f"GNN レコメンド: {uinfo['username'] if uinfo else uid} (top-{args.top_k})"
    print(f"  ノード: {G.number_of_nodes()}  エッジ: {G.number_of_edges()}")
    if args.static or args.output.endswith((".png",".jpg",".pdf")): visualize_matplotlib(G, args.output, title)
    else: visualize_pyvis(G, args.output, title)
    print(f"\n🎉 完了! → {args.output}")

if __name__ == "__main__": main()
