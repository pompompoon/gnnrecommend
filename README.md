# GNN ファッション レコメンドシステム

ヘテロジニアス Graph Neural Network (GAT v2) を用いたファッションアイテムの推薦システム。  
ユーザーの行動履歴（購入・閲覧・お気に入り）とカテゴリ階層構造をグラフとしてモデリングし、
BPR (Bayesian Personalized Ranking) で学習する。

## アーキテクチャ

```
PostgreSQL ──▶ GraphBuilder ──▶ HeteroData (PyG)
                                      │
                                HeteroGAT Encoder
                                (GATv2Conv × 3層)
                                      │
                              ┌───────┴───────┐
                              │               │
                         User Embed      Item Embed
                              │               │
                              └──── dot ──────┘
                                      │
                                  Score → Top-K
```

### ヘテロジニアスグラフ構造

| ノード        | 特徴量                                      |
|-------------|-------------------------------------------|
| user        | age, gender (one-hot), prefecture (one-hot) |
| item        | price, subcategory, brand, color, season    |
| subcategory | identity (one-hot)                         |
| category    | identity (one-hot)                         |

| エッジ (src → dst)                | 重み            |
|----------------------------------|----------------|
| user → purchased → item          | quantity       |
| user → viewed → item             | log(duration)  |
| user → favorited → item          | 1.0            |
| item → belongs_to → subcategory  | —              |
| subcategory → child_of → category| —              |
| + 全ての逆エッジ (ToUndirected)   |                |

## セットアップ

```bash
# 1. 依存パッケージ
pip install -r requirements.txt

# 2. PyTorch Geometric (環境に合わせて)
#    https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric

# 3. PostgreSQL データベース作成
createdb fashion_recommend

# 4. 全パイプライン実行 (データ生成 → 学習 → レコメンド)
python main.py --all

# 5. 個別コマンド
python main.py --generate-data
python main.py --train --epochs 100
python main.py --recommend --user-id 1 --top-k 10
python main.py --similar-items --item-id 42 --top-k 5
python main.py --save-recommendations
```

## カテゴリ構造 (画像準拠)

```
メンズ セール
├── レディトゥウェア
│   ├── Tシャツ
│   ├── シャツ＆ポロ
│   ├── スウェットシャツ
│   ├── ニット
│   ├── ジャケット、ボンバー
│   ├── アウター
│   ├── パンツ＆ショートパンツ
│   ├── デニム
│   ├── アクティブウェア
│   └── スイムウェア
├── シューズ
│   ├── スニーカー
│   ├── ブーツ＆レンジャー
│   ├── ローファー
│   └── サンダル
├── バッグ
│   ├── トートバッグ
│   ├── バックパック
│   ├── ショルダーバッグ
│   └── クラッチバッグ
└── アクセサリー
    ├── 革小物
    ├── ベルト
    ├── ファッションジュエリー
    └── ハット
```
