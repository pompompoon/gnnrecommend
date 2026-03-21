"""
PostgreSQL 接続・操作ユーティリティ

Windows 日本語環境対策 (2段構え):
  対策1: PGCLIENTENCODING=UTF8 で libpq の通信エンコーディングを強制
  対策2: APPDATA を ASCII パスに差し替え (pgpass.conf 読込時のcp932問題回避)
         Python / CRT / Win32 API の 3 レイヤー全てで設定
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

_orig_appdata: str | None = None

if sys.platform == "win32":
    import ctypes

    def _set_env_all_layers(name: str, value: str) -> None:
        os.environ[name] = value
        try:
            ctypes.cdll.msvcrt._putenv_s(name.encode("ascii"), value.encode("ascii"))
        except Exception:
            pass
        try:
            ctypes.windll.kernel32.SetEnvironmentVariableW(name, value)
        except Exception:
            pass

    _set_env_all_layers("PGCLIENTENCODING", "UTF8")
    _safe_dir = os.path.join(os.environ.get("SystemDrive", "C:"), os.sep, "pgtemp")
    os.makedirs(_safe_dir, exist_ok=True)
    _orig_appdata = os.environ.get("APPDATA")
    _set_env_all_layers("APPDATA", _safe_dir)
    _set_env_all_layers("PGPASSFILE", "NUL")
    _set_env_all_layers("PGSERVICEFILE", "NUL")
    _set_env_all_layers("PGSYSCONFDIR", _safe_dir)

import psycopg2
import psycopg2.extras

if sys.platform == "win32" and _orig_appdata:
    _set_env_all_layers("APPDATA", _orig_appdata)


class DatabaseManager:
    """PostgreSQL の接続とクエリ実行を管理"""

    def __init__(self, connect_kwargs: dict):
        self._kwargs = connect_kwargs

    @contextmanager
    def connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        _orig = None
        if sys.platform == "win32":
            _orig = os.environ.get("APPDATA", "")
            _safe = os.environ.get("PGSYSCONFDIR", "C:\\pgtemp")
            _set_env_all_layers("APPDATA", _safe)
        try:
            conn = psycopg2.connect(
                host=self._kwargs["host"],
                port=self._kwargs["port"],
                dbname=self._kwargs["dbname"],
                user=self._kwargs["user"],
                password=self._kwargs["password"],
                client_encoding="utf8",
            )
        finally:
            if _orig:
                _set_env_all_layers("APPDATA", _orig)
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
        with self.cursor(dict_cursor=False) as cur:
            psycopg2.extras.execute_values(cur, sql, values, page_size=page_size)

    def count(self, table: str) -> int:
        row = self.fetch_one(f"SELECT COUNT(*) AS cnt FROM {table}")
        return row["cnt"] if row else 0

    def init_schema(self, schema_path: str | Path = None) -> None:
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.sql"
        sql = Path(schema_path).read_text(encoding="utf-8")
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        print("  ✅ スキーマ初期化完了")

    def truncate_all(self) -> None:
        tables = [
            "recommendations", "favorites", "views", "purchases",
            "items", "users", "brands", "subcategories", "categories",
        ]
        with self.connection() as conn:
            with conn.cursor() as cur:
                for t in tables:
                    cur.execute(f"TRUNCATE TABLE {t} RESTART IDENTITY CASCADE")
        print("  🗑️  全テーブルを truncate しました（ID リセット済）")
