from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rs.llm.config import load_llm_config
from rs.utils.path_utils import resolve_from_repo_root


def _resolve_db_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return resolve_from_repo_root(raw).resolve()


def _fetch_rows(
        db_path: Path,
        *,
        memory_type: str | None,
        handler_substr: str | None,
        limit: int,
        order: str,
) -> list[dict[str, Any]]:
    order_sql = "updated_at_utc DESC, created_at_utc DESC" if order == "updated" else "created_at_utc ASC"
    query = f"""
        SELECT memory_id, namespace_json, memory_type, content, source_run_id,
               handler_name, created_at_utc, updated_at_utc, tags_json, kind
        FROM langmem_records
        WHERE 1=1
    """
    params: list[Any] = []
    if memory_type is not None:
        query += " AND memory_type = ?"
        params.append(memory_type)
    if handler_substr:
        query += " AND handler_name LIKE ?"
        params.append(f"%{handler_substr}%")
    query += f" ORDER BY {order_sql} LIMIT ?"
    params.append(limit)

    connect_uri = db_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(connect_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, params).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return []
            raise
    result: list[dict[str, Any]] = []
    for row in rows:
        namespace = json.loads(row["namespace_json"])
        tags = json.loads(row["tags_json"])
        result.append({
            "memory_id": row["memory_id"],
            "namespace": namespace,
            "memory_type": row["memory_type"],
            "content": row["content"],
            "source_run_id": row["source_run_id"],
            "handler_name": row["handler_name"],
            "created_at_utc": row["created_at_utc"],
            "updated_at_utc": row["updated_at_utc"],
            "tags": tags,
            "kind": row["kind"],
        })
    return result


def _count_by_type(db_path: Path) -> dict[str, int]:
    connect_uri = db_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(connect_uri, uri=True)) as connection:
        try:
            rows = connection.execute(
                "SELECT memory_type, COUNT(*) AS n FROM langmem_records GROUP BY memory_type"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise
    return {str(r[0]): int(r[1]) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect LangMem rows stored in the SQLite database (read-only).",
    )
    parser.add_argument(
        "--sqlite-path",
        default=None,
        help="Path to memory.sqlite3. Defaults to langmem_sqlite_path from LLM config / LANGMEM_SQLITE_PATH.",
    )
    parser.add_argument(
        "--type",
        dest="memory_type",
        choices=("all", "episodic", "semantic"),
        default="all",
        help="Filter by memory_type.",
    )
    parser.add_argument(
        "--handler",
        default=None,
        help="Substring filter on handler_name (SQL LIKE %%value%%).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to print after filtering.",
    )
    parser.add_argument(
        "--sort",
        choices=("created", "updated"),
        default="updated",
        help="Sort order: by created_at_utc ascending, or updated_at_utc descending (default).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON array to stdout.",
    )
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Only print counts per memory_type.",
    )
    args = parser.parse_args()

    cfg = load_llm_config()
    raw_path = args.sqlite_path or cfg.langmem_sqlite_path
    db_path = _resolve_db_path(raw_path)

    if not db_path.is_file():
        print(f"No database file at: {db_path}", file=sys.stderr)
        return 1

    if args.counts_only:
        counts = _count_by_type(db_path)
        if args.json:
            print(json.dumps(counts, indent=2))
        else:
            print(f"Database: {db_path}")
            if not counts:
                print("No langmem_records table or no rows.")
            else:
                for key in sorted(counts.keys()):
                    print(f"  {key}: {counts[key]}")
        return 0

    mem_filter: str | None = None if args.memory_type == "all" else args.memory_type
    rows = _fetch_rows(
        db_path,
        memory_type=mem_filter,
        handler_substr=args.handler,
        limit=max(1, args.limit),
        order=args.sort,
    )

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    counts = _count_by_type(db_path)
    print(f"Database: {db_path}")
    if counts:
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        print("Counts:", ", ".join(parts))
    print(f"Showing up to {args.limit} row(s)" + (f" (type={mem_filter})" if mem_filter else "") + ":")
    if not rows:
        print("  (no matching rows)")
        return 0

    for i, rec in enumerate(rows, start=1):
        ns = "/".join(str(x) for x in rec["namespace"])
        print("-" * 72)
        print(f"[{i}] {rec['memory_type']} | {rec['handler_name']} | id={rec['memory_id'][:12]}...")
        print(f"    namespace: {ns}")
        print(f"    run_id: {rec['source_run_id']}")
        print(f"    updated: {rec['updated_at_utc']}")
        content = str(rec["content"])
        preview = content if len(content) <= 500 else content[:497] + "..."
        print(f"    content: {preview}")
        if rec.get("tags"):
            print(f"    tags: {rec['tags']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
