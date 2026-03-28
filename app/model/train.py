import pandas as pd
import numpy as np
import joblib
from app.model.preprocessing import OutlierClipper
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE



#  Load Dataset

df = pd.read_csv("data/dataset.csv")


#  Feature Engineering
df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
df['age'] = df['age'] * (80 - 18) + 18
df['age'] = df['age'].astype(int)


#  Split Features & Target
X = df.drop("target", axis=1)
y = df["target"]


#  Column Types
categorical_cols = ["smoking_status"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]


#  Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


#  FULL PIPELINE
pipeline = Pipeline([
    ("outlier_clipping", OutlierClipper(numerical_cols)),
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", SVC(probability=True))
])


#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  Train Model
pipeline.fit(X_train, y_train)


#  Evaluate
preds = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f" Model Accuracy: {accuracy:.4f}")


#  Save Model (FULL PIPELINE)
joblib.dump(pipeline, "app/model/model.pkl")

print(" Pipeline (with scaling + SMOTE) saved successfully!")