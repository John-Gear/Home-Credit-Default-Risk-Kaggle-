from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from src.logger import get_logger
from src.preprocessor import load_data_train, load_data_test, build_train_test
import os
import json
import joblib

# инициируем логгирование
logger = get_logger('train')
logger.info('Training started')

X_train, y_train, X_test = build_train_test()

# Catboost preprocessing: categorial value
def fill_cat_nan(df, cat_cols):
    df=df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna('NA')
    return df

cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()

X_train = fill_cat_nan(X_train, cat_cols)
X_test  = fill_cat_nan(X_test, cat_cols)

# проверка перед обучением (в cat_cols не должны остатся nan)
if X_train[cat_cols].isna().sum().sum() == 0:
    logger.info('Кат. признаки в X_train подготовлены для обучения в CatBoost')
else:
    logger.error('Проверьте кат. признаки (X_train), найдены nan значения')

if X_test[cat_cols].isna().sum().sum() == 0:
    logger.info('Кат. признаки в X_test подготовлены')
else:
    logger.error('Проверьте кат. признаки (X_test), найдены nan значения')

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

# предсказания и метрики
train_pred = model.predict_proba(X_train)[:, 1]
roc_auc = roc_auc_score(y_train, train_pred)

metrics = {

}

logger.info(f'Metrics: \n%s', json.dumps(metrics, indent=4))

# сохранение метрик
os.makedirs('artefacts', exist_ok=True)

joblib.dump(model, 'artefacts/model.joblib')

with open('artefacts/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info('Model saved to artefacts')