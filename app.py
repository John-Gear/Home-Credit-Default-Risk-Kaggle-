from flask import Flask, request, jsonify
import pandas as pd
import json
from src.logger import get_logger
from src.predict import predict_proba, predict

logger = get_logger('api')

ID_COL = "SK_ID_CURR"

# получаем список колонок после обучения модели, при предсказании список колонок должен совпадать
with open('artefacts/expected_cols.json', 'r', encoding='utf-8') as f:
    EXPECTED_COLS = json.load(f)

app = Flask(__name__)
logger.info('API started')

# проверка жив ли сервер
@app.get('/health')
def health():
    return jsonify({'status': 'ok'})

# берем из json 1 клиента, возвращаем вероятность и предсказание
@app.post('/predict_single')
def predict_single():
    payload = request.get_json(silent=True)

    if payload is None or not isinstance(payload, dict) or len(payload) == 0:
        return jsonify({'error': 'ожидаем JSON с одним клиентом'}), 400
    
    df = pd.DataFrame([payload])

    missing = [c for c in EXPECTED_COLS if c not in df.columns] # валидация полученного датафрейма. Если колонки будут отличатся, то модель упадет
    if missing:
        return jsonify({'error': 'Потеряны колонки', 'missing': missing}), 400
    df = df[EXPECTED_COLS]

    proba = float(predict_proba(df)[0])
    pred = int(predict(df)[0])

    return jsonify({ID_COL: payload[ID_COL], 'Вероятности': proba, 'Предсказания': pred})

# берем из json список до 10000 клиентов, возвращаем вероятности и предсказания по каждому
@app.post('/predict_batch')
def predict_batch():
    payload = request.get_json(silent=True)

    if payload is None or not isinstance(payload, list) or len(payload) == 0:
        return jsonify({'error': 'ожидаем JSON с массивом клиентов'}), 400
    
    if len(payload) > 10000:
        return jsonify({'error': 'запрос должен быть до 10000 строк'}), 413

    df = pd.DataFrame(payload)

    missing = [c for c in EXPECTED_COLS if c not in df.columns] # валидация полученного датафрейма. Если колонки будут отличатся, то модель упадет
    if missing:
        return jsonify({'error': 'Потеряны колонки', 'missing': missing}), 400
    df = df[EXPECTED_COLS]

    proba = predict_proba(df)
    pred = predict(df)

    out = pd.Dataframe({ID_COL: df[ID_COL].values, 'Вероятности': proba, 'Предсказания': pred})

    return jsonify(out.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # дефолтный порт
