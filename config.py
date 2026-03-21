"""
プロジェクト全体の設定
"""
from dataclasses import dataclass, field


@dataclass
class DBConfig:
    """PostgreSQL 接続設定"""
    host: str = "localhost"
    port: int = 5432
    dbname: str = "fashion_recommend"
    user: str = "postgres"
    password: str = "postgres"

    @property
    def connect_kwargs(self) -> dict:
        """psycopg2.connect() に渡すキーワード引数"""
        return dict(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )


@dataclass
class DataConfig:
    """サンプルデータ生成の設定 (CPU 16GB 向け軽量版)"""
    n_users: int = 200          # 500 → 200
    n_items: int = 150          # 350 → 150
    n_purchases: int = 8000     # 25000 → 8000
    n_views: int = 15000        # 60000 → 15000
    n_favorites: int = 4000     # 12000 → 4000
    seed: int = 42
    # 密度: (8000+15000+4000) / (200×150) = 90%  ← 十分リッチ


@dataclass
class ModelConfig:
    """GNN モデルのハイパーパラメータ (CPU 軽量版)"""
    embedding_dim: int = 32
    hidden_channels: int = 64   # 128 → 64
    out_channels: int = 32      # 64 → 32
    num_layers: int = 2         # 3 → 2
    heads: int = 2              # 4 → 2
    dropout: float = 0.2
    learning_rate: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 100           # 200 → 100
    batch_size: int = 512       # 1024 → 512
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    early_stop_patience: int = 30  # 50 → 30
    device: str = "cpu"
    encoder_type: str = "gat"
    teleport_prob: float = 0.15
    num_iterations: int = 5     # 10 → 5 (APPNP 反復を半減)


@dataclass
class Config:
    """統合設定"""
    db: DBConfig = field(default_factory=DBConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
