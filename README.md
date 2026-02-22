# Home Credit Default Risk - Credit Scoring ML Pipeline (ML + API + Docker)

Проект решает задачу бинарной классификации по прогнозированию дефолта клиента и оптимизации портфельной прибыли банка на табличных данных **Home Credit**
**инженерный ML-пайплайн**: данные в SQLite, обучение модели `CatBoostClassifier`, сохранение артефактов, inference через Flask API и запуск в Docker.

---

## Ключевые идеи проекта

- **Источник данных — SQLite**
- **Модель сохранена как артефакт** `artefacts/model.joblib` (+ expected_cols.json колонки для валидации данных, metrics.json результаты обучения).
- Через docker.yml контейнеризацию **Inference отделён от обучения**: API стартует на артефактах полученных после обучения модели. Выдает предсказания по одному клиенту /predict_single или по списку /predict_batch (полученным из JSON)
- Обучение модели и работа Flask приложения **логируются через logger**
- Подготовка данных для обучения и инференса происходит через глобальный preprocessor (единый контракт данных + дополнительные проверки **снижают риск data leakage**)

---

## Структура проекта

```text
Home Credit Default Risk/
│
├── notebooks/
│       └── Home Credit notebook.ipynb        # исследовательский ноутбук
│
├── data/
│   ├── for_predict/
│   │   └── application_test.db
│   ├── application_train.db                 # SQLite база данных для обучения
│   └── HomeCredit_columns_description.csv   # текстовые описания колонок
│
├── artefacts/
│   ├── expected_cols.json                   # список входных фичей, ожидаемых api (контракт)
│   ├── model.joblib                         # сохранённый CatBoost model artifact (joblib)
│   └── metrics.json                         # метрики
│
├── src/
│   ├── db.py                                # загрузка данных из SQLite
│   ├── preprocessor.py                      # чистка/приведение типов, локальный препроцессор для CB
│   ├── train.py                             # обучение + сохранение model.joblib и metrics.json
│   ├── predict.py                           # порог THRESHOLD и вспомогательная логика inference
│   └── logger.py                            # логгер
│
├── app.py                                   # Flask API (/health, /predict_single, /predict_batch)
├── requirements.txt
├── docker-compose.yml                       # оркестрация контейнеров: 1 - обучение, 2 - flask api
└── Dockerfile
```

---

## Данные

В проекте используется датасет Home Credit Default Risk, загруженный в SQLite:

- файл: data/application_train.db для обучения модели
- файл: data/for_predict/application_test.db для предсказаний модели (имитация инференса)
- таблица: HomeCredit_columns_description.csv с описанием колонок в датасете application_train.db (нужно для исследоваительского ноутбука)
(взято: https://www.kaggle.com/competitions/home-credit-default-risk/overview)

TARGET:

* 0 — клиент не дефолтил
* 1 — клиент допустил дефолт

---

## Реализованный ML-пайплайн

В проекте реализованы следующие этапы:

### 1. EDA

* анализ распределения TARGET
* анализ числовых и категориальных признаков
* проверка пропусков
* анализ выбросов и т.п.

### 2. Data Preprocessing

* обработка пропусков
* подготовка категориальных признаков
* формирование обучающей и тестовой выборок
* финальная проверка размерности

---

### 3. Baseline: CatBoost

* построена базовая модель CatBoost
* первичная оценка ROC-AUC

---

### 4. Baseline Interpretation & Sanity Check

* анализ важности признаков
* проверка распределения предсказаний
* проверка логики модели

---

### 5. Advanced Models: LightGBM

* реализована альтернативная LightGBM-модель (в ноутбуке)
* оценка через cross-validation

---

### 6. Model Selection

* сравнение LightGBM с CatBoost
* выбор финальной архитектуры (CatBoost)
* обоснование выбора на основе CV-метрик

---

### 7. Hyperparameter Tuning (CatBoost)

* подбор гиперпараметров через K-Fold CV
* сравнение mean/std-auc

---

### 8. K-Fold Cross-Validation (OOF)

* генерация out-of-fold предсказаний
* честная оценка обобщающей способности модели

**OOF ROC-AUC ≈ 0.759**

---

### 9. Final CatBoost Training

* обучение финальной модели на всём train
* фиксация production-ready модели

---

### 10. Threshold Tuning

* подбор порогов Threshold
* анализ влияния порога на:

  * recall (перехват дефолтов)
  * approval rate (доля одобренных клиентов)
* построение зависимости recall ↔ approval
* построение матрицы распределений

---

### 11. Business Optimization

Реализована экономическая модель портфеля:

$$
E[profit] = (1 - p) \cdot profit - p \cdot (loan + profit)
$$

Где:

* p — предсказанная моделью вероятность дефолта
* E[profit] - ожидаемая прибыль на одного клиента
* loan - средний размер тела кредита в рублях
* profit = (loan * rate * (time_interval / 12)) - прибыль с одного хорошего клиента

Оптимальный threshold выбран по максимизации чистой прибыли.

Результат (пример для горизонта 12 месяцев):

* Optimal threshold: **0.13**
* Approval rate: **83%**
* Recall: **48%**
* Expected portfolio profit: **≈ 514 млн руб**


---

### 12. Inference (predict/API)

* Сохранение артефактов после обучения модели: artefacts/model.joblib, artefacts/metrics.json, artefacts/expected_cols.json
* Flask API Endpoints (get - /health, POST - /predict_single, /predict_batch)
* Docker (оркестрация контейнеров обучение -> flask api)

---

## Docker

1. Первичное обучение модели и получение артефактов
```bash
docker compose run --rm train
```

2. Запуск flask api
```bash
docker compose up api
```

---

## Ключевые идеи проекта

* В проекте реализован полный ML + Business pipeline для задачи кредитного скоринга
* ROC-AUC используется для выбора модели
* Threshold выбирается по конкертным экономическим показателям
* OOF необходим для честного ROC-AUC и как гарантия от переобучения/недообучения модели
* Модель переведена в денежный эквивалент: экономическая интерпретация в финансовых показателях + прогноз прибыли на новых клиентах (на имеющемся X_test)
* Инференс модели реализован через контейнеры docker. После обучения flask api готово к получению данных JSON для дальнейших предсказаний

---

## Используемые библиотеки

* pandas
* NumPy
* matplotlib / seaborn
* scikit-learn
* catboost
* lightgbm
* flask

---

## Визуализация
![Баланс между одобрением и риском](https://raw.githubusercontent.com/John-Gear/Home-Credit-Default-Risk-Kaggle-/refs/heads/main/%D0%91%D0%B0%D0%BB%D0%B0%D0%BD%D1%81%20%D0%BC%D0%B5%D0%B6%D0%B4%D1%83%20%D0%BE%D0%B4%D0%BE%D0%B1%D1%80%D0%B5%D0%BD%D0%B8%D0%B5%D0%BC%20%D0%B8%20%D1%80%D0%B8%D1%81%D0%BA%D0%BE%D0%BC.png)
![Матрица распределений](https://raw.githubusercontent.com/John-Gear/Home-Credit-Default-Risk-Kaggle-/refs/heads/main/%D0%9C%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0%20%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B9.png)

---

## Воспроизвести проект можно на Kaggle

https://www.kaggle.com/code/johngearonline/home-credit-default-risk-johngear
