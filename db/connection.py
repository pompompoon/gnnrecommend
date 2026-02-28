"""
PostgreSQL 接続・操作ユーティリティ
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import psycopg2
import psycopg2.extras


class DatabaseManager:
    """PostgreSQL の接続とクエリ実行を管理"""

    def __init__(self, connect_kwargs: dict):
        self._kwargs = connect_kwargs

    # --------------------------------------------------
    # Context managers
    # --------------------------------------------------
    @contextmanager
    def connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        conn = psycopg2.connect(**self._kwargs)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def cursor(self, dict_cursor: bool = True) -> Generator:
        factory = psycopg2.extras.RealDictCursor if dict_cursor else None
        with self.connection() as conn:
            cur = conn.cursor(cursor_factory=factory)
            try:
                yield cur
            finally:
                cur.close()

    # --------------------------------------------------
    # Query helpers
    # --------------------------------------------------
    def execute(self, sql: str, params: tuple | None = None) -> None:
        with self.cursor(dict_cursor=False) as cur:
            cur.execute(sql, params)

    def fetch_all(self, sql: str, params: tuple | None = None) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def fetch_one(self, sql: str, params: tuple | None = None) -> dict | None:
        with self.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def execute_values(self, sql: str, values: list[tuple], page_size: int = 2000) -> None:
        """psycopg2.extras.execute_values によるバルク INSERT"""
        with self.cursor(dict_cursor=False) as cur:
            psycopg2.extras.execute_values(cur, sql, values, page_size=page_size)

    def count(self, table: str) -> int:
        row = self.fetch_one(f"SELECT COUNT(*) AS cnt FROM {table}")
        return row["cnt"] if row else 0

    # --------------------------------------------------
    # Schema management
    # --------------------------------------------------
    def init_schema(self, schema_path: str | Path = None) -> None:
        """SQL ファイルからスキーマを初期化"""
        if schema_path is None:
            # main.py の実行位置に依存しないよう、このファイル基準で解決
            schema_path = Path(__file__).parent / "schema.sql"
        sql = Path(schema_path).read_text(encoding="utf-8")
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        print("  ✅ スキーマ初期化完了")

    def truncate_all(self) -> None:
        """全テーブルを TRUNCATE（SERIAL のシーケンスもリセット）"""
        tables = [
            "recommendations", "favorites", "views", "purchases",
            "items", "users", "brands", "subcategories", "categories",
        ]
        with self.connection() as conn:
            with conn.cursor() as cur:
                for t in tables:
                    cur.execute(f"TRUNCATE TABLE {t} RESTART IDENTITY CASCADE")
        print("  🗑️  全テーブルを truncate しました（ID リセット済）")
