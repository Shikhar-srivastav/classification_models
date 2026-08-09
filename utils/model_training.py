import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from constants.common import target_cols

train_df = pd.read_csv("train_data.csv")

fault_cols = target_cols
feature_cols = [c for c in train_df.columns if c not in fault_cols]

y_train = train_df[fault_cols].idxmax(axis=1)
X_train = train_df[feature_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

scaler_file = "models/standard_scaler.pkl"
joblib.dump(scaler, scaler_file, compress=3)
print(f"Saved {scaler_file}")

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    file_name = f"models/{name}.pkl"
    joblib.dump(model, file_name, compress=3)
    print(f"Saved {file_name}")