"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

paths.py
system shared paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

IMAGE_PATH: Path = PROJECT_ROOT / "data" / "images"
IMAGE_PATH.mkdir(parents=True, exist_ok=True)

DB_PATH: Path = PROJECT_ROOT / "data" / "db"
DB_PATH.mkdir(parents=True, exist_ok=True)