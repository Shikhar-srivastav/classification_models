import pandas as pd
import joblib

from sklearn import metrics

test_df = pd.read_csv("train_data.csv")
models = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest",
]

fault_cols = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
feature_cols = [c for c in test_df.columns if c not in fault_cols]

y_test = test_df[fault_cols].idxmax(axis=1)
X_test = test_df[feature_cols]

scaler_file = "models/standard_scaler.pkl"
scaler = joblib.load(scaler_file)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(name):
    if (name not in models):
        return
    
    file_name = f"models/{name}.pkl"
    model = joblib.load(file_name)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)

    model_metrics = {
        "Model": name,
        "Accuracy": metrics.accuracy_score(y_test, predictions),
        "AUC": metrics.roc_auc_score(y_test, probabilities, multi_class="ovr"),
        "Precision": metrics.precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": metrics.recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1": metrics.f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": metrics.matthews_corrcoef(y_test, predictions),
    }

    return model_metrics


