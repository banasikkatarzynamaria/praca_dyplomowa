# Breast Cancer Detection Project \- Katarzyna Banasik

This project aims to build a machine learning model to detect malignant tumors using the Breast Cancer Wisconsin dataset. Below is an overview and results of the project's methodology and findings, along with deployment instructions for the API.

### **Problem Definition**

Breast cancer detection is critical for early diagnosis and treatment. The primary objective of this project is to develop a model with a high recall for malignant cases, minimizing false negatives, which is crucial in a clinical setting.

### **Data Preprocessing**

* **Dataset**: Breast Cancer Wisconsin dataset, preprocessed with PCA for dimensionality reduction, data normalization and feature dropping based on the feature importance.  
* **Feature Engineering**: PCA was applied on correlated feature groups.  
* **Dataset Used**: Breast Cancer Wisconsin (sklearn) dataset preprocessed as explained above and renamed `X_combined` post preprocessing.

### **Model Training and Selection**

A selection of models was trained and evaluated using cross-validation:

* **Models Compared**: Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN.  
* **Optimization Goal**: Maximize recall for malignant cases.  
* **Tools Used**: `scikit-learn` for model development and `mlflow` for experiment tracking.

### **Results**

* **Selected Model**: Random Forest, offering the highest recall while not being the most energy-consuming.&nbsp;  
* **Performance Metrics**:  
  * **Malignant Class**: Precision 0.98, Recall 0.93, F1 Score 0.95.  
  * **Benign Class**: Precision 0.96, Recall 0.99, F1 Score 0.97.  
* **Receiver Operating Characteristic (ROC)**: AUC of 1.00, suggesting high model discrimination capability.  
* **Precision-Recall Curve**: Highlighting high recall and precision until over-fitting and dropping the precision for too high recall. The 97% recall which we achieved seems high enough and not too high to have an overfit model.&nbsp;

### **Deployment**

The model is deployed as a FastAPI application, enabling easy integration with other systems for real-time predictions.

### **Setup & Deployment**

Follow these guides for local setup, Docker containerization, and cloud deployment (Google Cloud Run).

# Breast Cancer Detection API Deployment Guide

This repository contains a FastAPI application to serve the Breast Cancer Detection model optimized for high recall on malignant cases.

## Project Structure

- `cancer-detection-api.py`: The FastAPI application.  
- `model.joblib`: The trained Random Forest model (ensure this is exported from your notebook).  
- `Dockerfile`: Containerization configuration.  
- `api-test-script.py`: A Python script to verify the endpoint.  
- `requirements.txt`: Python dependencies.

## Setup & Local Run

1. **Install Dependencies**:

```sh
pip install fastapi uvicorn joblib pandas scikit-learn requests
```

2. **Run API**:

```sh
uvicorn cancer-detection-api:app --reload --port 8080
```

3. **Test API**: Open a new terminal and run:

```sh
python api-test-script.py
```

## Docker Deployment

1. **Build Image**:

```sh
docker build -t cancer-api .
```

2. **Run Container**:

```sh
docker run -p 8080:8080 cancer-api
```

## Cloud Deployment

### Google Cloud Run

1. **Submit Build**: `gcloud builds submit --tag gcr.io/[PROJECT-ID]/cancer-api`  
2. **Deploy**: `gcloud run deploy --image gcr.io/[PROJECT-ID]/cancer-api --platform managed`

### Heroku

1. **Login**: `heroku login`  
2. **Create App**: `heroku create [app-name]`  
3. **Container Push**: `heroku container:push web -a [app-name]`  
4. **Container Release**: `heroku container:release web -a [app-name]`

## API Endpoints

- `GET /`: Health check.  
- `POST /predict`: Submit feature data to receive classification (Malignant/Benign).