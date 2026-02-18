# Home Credit Default Risk - Credit Scoring ML Pipeline

Задача бинарной классификации по прогнозированию дефолта клиента и оптимизации портфельной прибыли банка.

---

## Обзор проекта

Проект посвящён построению модели кредитного скоринга на основе табличных данных (Kaggle: *Home Credit Default Risk* - https://www.kaggle.com/competitions/home-credit-default-risk/overview).

Цель проекта:

* построить ML-пайплайн для кредитного скоринга
* показать корректную валидацию модели (K-Fold, OOF)
* реализовать выбор порога принятия решения (threshold tuning)
* перевести модель в экономическую модель прибыли банка

Проект охватывает как техническую часть (модель), так и бизнес-слой.

---

## Данные

* `application_train.csv` — исторические клиенты с известным TARGET (дефолт / не дефолт)
* `application_test.csv` — новые клиенты без TARGET
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

* реализована альтернативная LightGBM-модель
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

[
E[profit] = (1 - p) \cdot profit - p \cdot (loan + profit)
]

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

## Ключевые идеи проекта

* ROC-AUC используется для выбора модели
* Threshold выбирается по конкертным экономическим показателям
* OOF необходим для честного ROC-AUC и как гарантия от переобучения/недообучения модели
* Модель переведена в денежный эквивалент

---

## Используемые библиотеки

* pandas
* NumPy
* matplotlib / seaborn
* scikit-learn
* catboost
* lightgbm

---

## Визуализация
![Баланс между одобрением и риском](https://github.com/John-Gear/Home-Credit-Default-Risk-Kaggle-/blob/main/%D0%91%D0%B5%D0%B7%20%D0%BD%D0%B0%D0%B7%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F%20(1).png)
![Матрица распределений](https://raw.githubusercontent.com/John-Gear/Home-Credit-Default-Risk-Kaggle-/refs/heads/main/%D0%91%D0%B5%D0%B7%20%D0%BD%D0%B0%D0%B7%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F.png)

---

## Структура проекта

```
notebooks/
    Home Credit notebook.ipynb

src/
```

---

## Как запустить

```bash
pip install -r requirements.txt
Home Credit notebook.ipynb
```

---

## Итог

В проекте реализован полный ML + Business pipeline для задачи кредитного скоринга:

* корректная валидация полученных данных
* выбор модели (+ сравнение)
* выбор порога отказы/одобрения (threshold)
* экономическая интерпретация в финансовых показателях
* прогноз прибыли на новых клиентах (на имеющемся X_test)

---
