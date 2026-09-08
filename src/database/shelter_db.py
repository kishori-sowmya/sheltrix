"""
Module 15: SQLite Database Persistence Layer
Stores climate profiles, materials, candidate shelter designs, training datasets,
surrogate model metadata, optimization runs, and validation reports.
"""

import sqlite3
import json
import os
from typing import Dict, Any, List, Optional
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "shelter.db")


class ShelterDatabase:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Climate Profiles Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS climates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL,
                season TEXT NOT NULL,
                data_source TEXT NOT NULL,
                is_synthetic INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Materials Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS materials (
                key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                source TEXT,
                cost_base REAL
            )
            """)

            # Datasets Metadata Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_metadata (
                version TEXT PRIMARY KEY,
                n_samples INTEGER NOT NULL,
                climate_scenarios TEXT NOT NULL,
                seed INTEGER NOT NULL,
                filepath TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Model Runs & Optimization Experiments Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                run_id TEXT PRIMARY KEY,
                climate_scenario TEXT NOT NULL,
                pop_size INTEGER NOT NULL,
                n_gen INTEGER NOT NULL,
                n_pareto_designs INTEGER NOT NULL,
                elapsed_sec REAL NOT NULL,
                pareto_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Validation Reports Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                top_rank INTEGER NOT NULL,
                surrogate_heating_kWh REAL,
                physics_heating_kWh REAL,
                ansys_heating_kWh REAL,
                error_pct REAL,
                ansys_status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.commit()

    def save_climate(self, key: str, name: str, region: str, season: str, data_source: str, is_synthetic: bool, summary: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.cursor().execute("""
            INSERT OR REPLACE INTO climates (id, name, region, season, data_source, is_synthetic, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (key, name, region, season, data_source, int(is_synthetic), json.dumps(summary)))
            conn.commit()

    def log_optimization_run(self, run_id: str, scenario: str, pop_size: int, n_gen: int, pareto_df: pd.DataFrame, elapsed: float):
        with self.get_connection() as conn:
            conn.cursor().execute("""
            INSERT INTO optimization_runs (run_id, climate_scenario, pop_size, n_gen, n_pareto_designs, elapsed_sec, pareto_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run_id, scenario, pop_size, n_gen, len(pareto_df), elapsed, pareto_df.to_json(orient="records")))
            conn.commit()

    def log_validation(self, run_id: str, rank: int, surr_h: float, phys_h: float, ansys_h: Optional[float], error: float, status: str):
        with self.get_connection() as conn:
            conn.cursor().execute("""
            INSERT INTO validation_reports (run_id, top_rank, surrogate_heating_kWh, physics_heating_kWh, ansys_heating_kWh, error_pct, ansys_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run_id, rank, surr_h, phys_h, ansys_h, error, status))
            conn.commit()


if __name__ == "__main__":
    db = ShelterDatabase()
    print("Database initialized successfully at:", DB_PATH)
