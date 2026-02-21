import joblib
from src.preprocessor import preprocessing, prepare_for_catboost
import numpy as np
import pandas as pd
import os

MODEL_PATH = 'artifacts/model.joblib'
THRESHOLD = 0.13 # порог берем из исследований ранее проведенных в Home Credit notebook.ipynb

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Модель не найдена в {MODEL_PATH}, сначала запустите train.py')
    return joblib.load(MODEL_PATH)

# предсказываем вероятности
def predict_proba(df_features: pd.DataFrame) -> np.ndarray:
    model = load_model()

    df = preprocessing(df_features)
    df, _cat_cols = prepare_for_catboost(df) # cat_cols не нужен для predict

    probs = model.predict_proba(df)[:, 1]
    return probs

# предсказываем класс 0/1
def predict(df_features: pd.DataFrame, threshold: float = THRESHOLD) -> np.ndarray:
    probs = predict_proba(df_features)
    preds = (probs >= threshold).astype(int)
    return preds