import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ========================
# Load data
# ========================
df = pd.read_csv("data/dataset.csv")



# ========================
# Features & Target
# ========================
selected_features = [
    "age",
    "bmi",
    "glucose_level",
    "blood_pressure",
    "smoking_status"
]

X = df[selected_features]
y = df["target"]


# ========================
# Column separation
# ========================
categorical_cols = ["smoking_status"]
numerical_cols = [col for col in selected_features if col not in categorical_cols]


# ========================
# Preprocessing
# ========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


# ========================
# Train-Test Split
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ========================
# Models (8+)
# ========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(),

    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),

    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),

    "SVM": SVC(probability=True),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes": GaussianNB()
}


# ========================
# Evaluation Function
# ========================
def evaluate_model(pipeline, X_test, y_test):
    probs = pipeline.predict_proba(X_test)[:, 1]

    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.3, 0.7, 0.05):
        preds = (probs > t).astype(int)
        report = classification_report(y_test, preds, output_dict=True)
        f1 = report["1"]["f1-score"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    preds = (probs > best_threshold).astype(int)

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    return best_threshold, best_f1, accuracy, roc_auc


# ========================
# Train & Compare
# ========================
results = []

for name, model in models.items():
    print(f"\n========================")
    print(f"Model: {name}")
    print(f"========================")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    threshold, f1, acc, roc = evaluate_model(pipeline, X_test, y_test)

    print(f"Best Threshold: {threshold:.2f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    results.append({
        "model": name,
        "f1": f1,
        "accuracy": acc,
        "roc_auc": roc,
        "threshold": threshold
    })


# ========================
# Pick Best Model
# ========================
results_df = pd.DataFrame(results)

best_model = results_df.sort_values(by="f1", ascending=False).iloc[0]

print("\n========================")
print(" BEST MODEL")
print("========================")
print(best_model)