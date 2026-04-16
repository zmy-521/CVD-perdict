# ==============================================================================
# Final Model Training and Serialization Pipeline
# Architecture: Dual-Track XGBoost Models (Gatekeeper, Track A, Track B, Global)
# ==============================================================================

import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# ================= 1. Configuration of File Paths =================
# Define relative paths for open-source repository
DATA_DIR = "./data/"
OUTPUT_DIR = "./models/"  # Centralized directory for serialized models

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Note: Ensure these specific derivation datasets are placed in the DATA_DIR
path_gatekeeper = os.path.join(DATA_DIR, "dataset_gatekeeper.xlsx")
path_track_a = os.path.join(DATA_DIR, "dataset_track_a.xlsx")
path_track_b = os.path.join(DATA_DIR, "dataset_track_b.xlsx")
path_global = os.path.join(DATA_DIR, "dataset_global.xlsx")

# ================= 2. Model Specific Configurations =================
models_config = {
    'Gatekeeper': {
        'file_path': path_gatekeeper,
        'features': ['Age', 'BUN', 'RDW', 'SUA', 'HbA1c', 'Cl', 'A/G', 'NEU#'],
        'target': 'ODKD_Label'
    },
    'Track_A': {
        'file_path': path_track_a,
        'features': ['ALT', 'Age', 'RDW', 'Non-HDL-C', 'PLT', 'HbA1c', 'Cl', 'SCr'],
        'target': 'CVD_Label'
    },
    'Track_B': {
        'file_path': path_track_b,
        'features': ['SUA', 'Age', 'K', 'RDW', 'Non-HDL-C', 'MCV', 'SCr'],
        'target': 'CVD_Label'
    },
    'Global': {
        'file_path': path_global,
        'features': ['SUA', 'ALT', 'MON#', 'Age', 'LYM#', 'RDW', 'Non-HDL-C', 'K', 'PLT', 'MCV', 'SCr'],
        'target': 'CVD_Label'
    }
}

# ================= 3. Model Training and Serialization =================
if __name__ == "__main__":
    print("-" * 60)
    print("Initiating Final Model Training and Serialization Pipeline...")

    for model_name, config in models_config.items():
        print(f"\nProcessing Model: {model_name} ...")

        if not os.path.exists(config['file_path']):
            print(f"  [Error] Dataset not found: {config['file_path']}. Please check the data directory. Skipping...")
            continue

        # Load specific derivation subset
        df = pd.read_excel(config['file_path'])

        # Extract specific features and target variable
        X = df[config['features']]
        y = df[config['target']]

        # Calculate dynamic class imbalance ratio for optimal scale_pos_weight
        neg_count = len(y[y == 0])
        pos_count = len(y[y == 1])
        imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        print(
            f"  => Class Distribution: Negative {neg_count}, Positive {pos_count} | scale_pos_weight: {imbalance_ratio:.2f}")

        # Construct optimized XGBoost Pipeline
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('xgb', XGBClassifier(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=imbalance_ratio,
                eval_metric='auc',
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Execute Model Fitting
        print("  => Fitting model to the derivation cohort...")
        xgb_pipeline.fit(X, y)

        # Serialize and Save Pipeline as .pkl
        save_path = os.path.join(OUTPUT_DIR, f"Model_{model_name}.pkl")
        joblib.dump(xgb_pipeline, save_path)
        print(f"  => Success! Model serialized and exported to: {save_path}")

    print("\nPipeline execution completed. All specified models have been trained and exported.")
    print("-" * 60)