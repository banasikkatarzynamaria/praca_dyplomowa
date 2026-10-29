
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cancer_detection_api.py .
COPY best_model.joblib .

EXPOSE 8080

CMD ["uvicorn", "cancer_detection_api.py:app", "--host", "0.0.0.0", "--port", "8080"]
