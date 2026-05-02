"""
One-shot training script — generates data, trains both models.
Run: python -m src.models.trainer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generator import generate
from src.data.preprocessor import load_and_prepare
from src.models import risk_model, fraud_model


def train_all():
    print("Generating synthetic data...")
    generate(save=True)

    print("\nPreparing features...")
    X_train, X_test, y_train, y_test, scaler, df = load_and_prepare()

    print("\nTraining risk model...")
    risk_model.train(X_train, y_train, X_test, y_test)

    print("\nTraining fraud model...")
    fraud_model.train(X_train, y_train, X_test, y_test)

    print("\nAll models trained and saved.")


if __name__ == "__main__":
    train_all()
