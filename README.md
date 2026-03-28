#  Health Risk Prediction API

**Production-Ready Machine Learning Pipeline with FastAPI & Docker**



##  Overview

This project demonstrates an end-to-end **production-grade Machine Learning pipeline**, transitioning from exploratory analysis to a fully deployable API service.

The system predicts health risk using a **Support Vector Machine (SVM)** model trained on structured medical data, with proper handling of preprocessing, outliers, and class imbalance.

The project reflects **real-world ML engineering practices**, including modular architecture, reproducible pipelines, and containerized deployment.



##  Key Features

* **End-to-End ML Pipeline**

  * Feature engineering
  * Outlier handling (IQR-based clipping)
  * Scaling & encoding
  * Class imbalance handling using SMOTE
  * Model training using SVM

* **Production-Ready Design**

  * Single serialized pipeline (`joblib`)
  * No preprocessing mismatch between training and inference
  * Modular and scalable codebase

* **FastAPI Backend**

  * RESTful `/predict` endpoint
  * Input validation using Pydantic
  * Interactive API docs (Swagger UI)

* **Containerization**

  * Dockerized application
  * Ready for cloud deployment

---

##  Project Structure

```
health-ml-api/
│
├── app/
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Request schema
│   └── model/
│       ├── train.py         # Training pipeline
│       ├── predict.py       # Inference logic
│       ├── model.pkl        # Serialized pipeline
│       └── custom_transformers.py  # Custom preprocessing
│
├── data/
│   └── dataset.csv          # Input dataset
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

##  Machine Learning Pipeline

The model pipeline is built using `scikit-learn` and `imblearn`:

```
Raw Data
   ↓
Outlier Clipping (Custom Transformer)
   ↓
Standard Scaling (Numerical Features)
   ↓
One-Hot Encoding (Categorical Features)
   ↓
SMOTE (Class Imbalance Handling)
   ↓
SVM Classifier
```

### Highlights:

* **Custom Transformer** ensures reusable preprocessing logic
* **ColumnTransformer** handles mixed data types
* **SMOTE** applied only during training to prevent data leakage
* Entire pipeline serialized for consistent inference

---

##  Running the Project Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python app/model/train.py
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

### 4. Access API Docs

```
http://127.0.0.1:8000/docs
```

---

##  API Usage

### Endpoint

```
POST /predict
```



##  Docker Setup

### Build Image

```bash
docker build -t health-ml-api .
```

### Run Container

```bash
docker run -p 8000:8000 health-ml-api
```

---


###  Start Command (FastAPI)

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```



##  Learnings & Concepts Covered

* Transition from notebook → production ML system
* Building reusable preprocessing pipelines
* Avoiding data leakage in ML workflows
* Handling imbalanced datasets (SMOTE)
* Designing inference-ready APIs
* Containerizing ML applications with Docker

---

##  Future Improvements

* Model monitoring & logging
* CI/CD integration (GitHub Actions)
* Frontend dashboard for predictions
* Model versioning and experiment tracking
* Deployment with autoscaling (Kubernetes / Fly.io)

---

**Mohan**
Aspiring AI Engineer focused on building production-grade ML systems.
