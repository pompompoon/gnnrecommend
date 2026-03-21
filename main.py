#!/usr/bin/env python3
"""GNN Fashion Recommendation System"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from config import Config
from data.generate_sample_data import generate_all
from models.graph_builder import GraphBuilder
from models.gnn_model import GNNRecommender
from models.train import Trainer
from models.recommend import Recommender

def parse_args():
    p = argparse.ArgumentParser(description="GNN Fashion Recommender")
    p.add_argument("--generate-data", action="store_true"); p.add_argument("--train", action="store_true")
    p.add_argument("--recommend", action="store_true"); p.add_argument("--similar-items", action="store_true")
    p.add_argument("--save-recommendations", action="store_true"); p.add_argument("--all", action="store_true")
    p.add_argument("--user-id", type=int, default=1); p.add_argument("--item-id", type=int, default=1)
    p.add_argument("--top-k", type=int, default=10); p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--encoder", choices=["gat", "appnp"], default="gat")
    p.add_argument("--teleport", type=float, default=0.15); p.add_argument("--ppr-iters", type=int, default=10)
    p.add_argument("--db-host", default="localhost"); p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--db-name", default="fashion_recommend"); p.add_argument("--db-user", default="postgres")
    p.add_argument("--db-password", default="postgres")
    return p.parse_args()

def make_config(args):
    cfg = Config()
    cfg.db.host = args.db_host; cfg.db.port = args.db_port; cfg.db.dbname = args.db_name
    cfg.db.user = args.db_user; cfg.db.password = args.db_password
    if args.epochs is not None: cfg.model.epochs = args.epochs
    cfg.model.encoder_type = args.encoder; cfg.model.teleport_prob = args.teleport; cfg.model.num_iterations = args.ppr_iters
    if torch.cuda.is_available(): cfg.model.device = "cuda"
    return cfg

def build_graph_and_model(cfg):
    print("\n📐 グラフ構築中 …")
    builder = GraphBuilder(cfg.db.connect_kwargs); data = builder.build()
    in_ch = {nt: data[nt].x.shape[1] for nt in data.node_types}; mc = cfg.model
    enc = mc.encoder_type.upper()
    if mc.encoder_type == "appnp":
        print(f"🧠 モデル初期化  encoder={enc}  α={mc.teleport_prob}  K={mc.num_iterations}  hidden={mc.hidden_channels}  out={mc.out_channels}")
    else:
        print(f"🧠 モデル初期化  encoder={enc}  hidden={mc.hidden_channels}  out={mc.out_channels}  layers={mc.num_layers}  heads={mc.heads}")
    model = GNNRecommender(metadata=data.metadata(), in_channels_dict=in_ch, hidden_channels=mc.hidden_channels, out_channels=mc.out_channels, num_layers=mc.num_layers, heads=mc.heads, dropout=mc.dropout, encoder_type=mc.encoder_type, teleport_prob=mc.teleport_prob, num_iterations=mc.num_iterations)
    print(f"  パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    return model, data, builder

def load_checkpoint(model, device, path="checkpoints/best_model.pt"):
    ckpt = Path(path)
    if ckpt.exists(): model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)); print(f"  ✅ チェックポイント読込: {ckpt}")
    else: print(f"  ⚠️  チェックポイント未検出 ({ckpt})")

def main():
    args = parse_args(); cfg = make_config(args)
    print("=" * 72); print("  GNN Fashion Recommendation System"); print("=" * 72)
    if args.generate_data or args.all: generate_all(cfg)
    if args.train or args.all:
        model, data, builder = build_graph_and_model(cfg); Trainer(model, data, cfg.model).train(epochs=args.epochs)
    if args.recommend or args.all:
        model, data, builder = build_graph_and_model(cfg); load_checkpoint(model, cfg.model.device)
        rec = Recommender(model, builder, data, cfg)
        if args.all:
            from db.connection import DatabaseManager
            db = DatabaseManager(cfg.db.connect_kwargs)
            for row in db.fetch_all("SELECT user_id FROM users ORDER BY user_id LIMIT 3"): rec.print_user_recommendations(row["user_id"], args.top_k)
        else: rec.print_user_recommendations(args.user_id, args.top_k)
    if args.similar_items:
        model, data, builder = build_graph_and_model(cfg); load_checkpoint(model, cfg.model.device)
        rec = Recommender(model, builder, data, cfg); items = rec.find_similar_items(args.item_id, args.top_k)
        if items:
            print(f"\n  🎯 類似アイテム Top-{args.top_k}:")
            for it in items: print(f"     {it['rank']:2d}. {it['name']}  [{it['category']}>{it['subcategory']}]  ¥{it['price']:,}  sim={it['similarity']:.4f}")
    if args.save_recommendations:
        model, data, builder = build_graph_and_model(cfg); load_checkpoint(model, cfg.model.device)
        Recommender(model, builder, data, cfg).save_to_db(top_k=20)
    if not any([args.generate_data, args.train, args.recommend, args.similar_items, args.save_recommendations, args.all]):
        print("\n  python main.py --all --db-password パスワード\n  python main.py --all --encoder appnp --db-password パスワード\n  python main.py --help")

if __name__ == "__main__": main()
