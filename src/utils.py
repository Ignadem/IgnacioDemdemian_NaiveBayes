from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def load_default_dataset() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    features["target"] = dataset.target
    return features


def save_model(model, filename: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / filename
    joblib.dump(model, output_path)
    return output_path
