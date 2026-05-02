"""
One-time setup: install deps, train models, build FAISS index.
Run: python setup.py
"""

import subprocess
import sys


def run(cmd: str):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("=" * 60)
    print("AFDIS — One-Time Setup")
    print("=" * 60)

    run("pip install -r requirements.txt")

    print("\nTraining ML models (generates synthetic data first)...")
    run("python -m src.models.trainer")

    print("\nBuilding FAISS embedding index...")
    run("python -m src.rag.embeddings")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("Start API:       python run_api.py")
    print("Start Dashboard: python run_dashboard.py")
    print("=" * 60)
