# контейнер для запуска flask api (предварительно надо обучить модель на тестовых данных, получить artefacts в model.joblib)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r -requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]