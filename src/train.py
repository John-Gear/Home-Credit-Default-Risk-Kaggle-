from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from src.logger import get_logger
from src.preprocessor import prepare_for_catboost, build_train_test
import os
import json
import joblib

# инициируем логгирование
logger = get_logger('train')
logger.info('Training started')

X_train, y_train, _ = build_train_test()

X_train, cat_cols = prepare_for_catboost(X_train)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=False
)

model.fit(X_train, y_train, cat_features=cat_cols)

train_proba = model.predict_proba(X_train)[:, 1]
roc_auc_train = roc_auc_score(y_train, train_proba)

metrics = {
    'roc_auc_train': float(roc_auc_train),
    'model': 'CatBoostClassifier',
    'train_rows': int(X_train.shape[0]),
    'n_features': int(X_train.shape[1]),
    'cat_cols_count': int(len(cat_cols)),
}

logger.info(f'Metrics: \n%s', json.dumps(metrics, indent=4))

# сохранение метрик
os.makedirs('artefacts', exist_ok=True)

joblib.dump(model, 'artefacts/model.joblib')

with open('artefacts/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info('Model saved to artefacts')