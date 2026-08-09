import pandas as pd
import joblib

from constants.common import model_options, target_cols

models = model_options.values()

def model_predictions(model_name, test_df):
    fault_cols = target_cols
    feature_cols = [c for c in test_df.columns if c not in fault_cols]

    y_test = test_df[fault_cols].idxmax(axis=1)
    X_test = test_df[feature_cols]

    scaler_file = "models/standard_scaler.pkl"
    scaler = joblib.load(scaler_file)
    X_test_scaled = scaler.transform(X_test)

    if (model_name not in models):
        return
        
    file_name = f"models/{model_name}.pkl"
    model = joblib.load(file_name)
    
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)

    return y_test, predictions, probabilities