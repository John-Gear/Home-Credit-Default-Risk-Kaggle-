import pandas as pd
from src.db import read_sql_test
from src.predict import predict_proba, predict

# предсказываем вероятности/классы на реальных данных
df_test = read_sql_test("SELECT * FROM application_test")
proba = predict_proba(df_test)
pred  = predict(df_test)

# собираем таблицу SK_ID_CURR + proba + pred
out = df_test[["SK_ID_CURR"]].copy()
out["proba"] = proba.values
out["pred"] = pred.values

# сохраняем
out.to_csv("artefacts/test_predictions.csv", index=False)