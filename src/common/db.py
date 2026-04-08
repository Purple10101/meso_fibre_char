"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260408

db.py
shared database module for SS4 and SS5 to collect per-image results.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import sqlite3
import json
import os

from src.common.paths import DB_PATH
DB_FILE = DB_PATH / "results.db"



def _connect():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ss4_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            mesh_id INTEGER NOT NULL,
            length_mm REAL,
            width_mm REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ss5_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            result_data TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def write_ss4_results(image_id, fibres):
    """
    Write SS4 fibre measurements.
    fibres: list of dicts with keys mesh_id, length_mm, width_mm
    """
    conn = _connect()
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO ss4_results (image_id, mesh_id, length_mm, width_mm) VALUES (?, ?, ?, ?)",
        [(image_id, f["mesh_id"], f["length_mm"], f["width_mm"]) for f in fibres]
    )
    conn.commit()
    conn.close()


def write_ss5_results(image_id, result_data):
    """
    Write SS5 modelling results.
    result_data: dict (stored as JSON)
    """
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO ss5_results (image_id, result_data) VALUES (?, ?)",
        (image_id, json.dumps(result_data))
    )
    conn.commit()
    conn.close()


def read_image_results(image_id):
    """
    Read all data for a given image. SS4 only.
    Returns dict with ss4 and ss5 results.
    """
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("SELECT mesh_id, length_mm, width_mm FROM ss4_results WHERE image_id = ?", (image_id,))
    ss4 = [dict(row) for row in cursor.fetchall()]

    cursor.execute("SELECT result_data FROM ss5_results WHERE image_id = ?", (image_id,))
    row = cursor.fetchone()
    ss5 = json.loads(row["result_data"]) if row else None

    conn.close()

    return {
        "image_id": image_id,
        "fibres": ss4,
        "modelling": ss5
    }