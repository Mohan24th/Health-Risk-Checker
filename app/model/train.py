import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/dataset.csv")


# =========================
# SELECT CLEAN FEATURES
# =========================
selected_features = [
    "age",
    "bmi",
    "glucose_level",
    "blood_pressure",
    "smoking_status"
]

X = df[selected_features]
y = df["target"]


# =========================
# COLUMN TYPES
# =========================
categorical_cols = ["smoking_status"]
numerical_cols = [col for col in selected_features if col not in categorical_cols]


# =========================
# PREPROCESSING
# =========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


# =========================
# PIPELINE (CLEAN)
# =========================
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", SVC(probability=True))
])


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# TRAIN MODEL
# =========================
pipeline.fit(X_train, y_train)


# =========================
# EVALUATE
# =========================
preds = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Model Accuracy: {accuracy:.4f}")


# =========================
# SAVE MODEL
# =========================
joblib.dump(pipeline, "app/model/model.pkl")

print("Clean model saved successfully!")