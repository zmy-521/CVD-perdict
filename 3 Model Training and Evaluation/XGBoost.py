# ==============================================================================
# Machine Learning Model Optimization and Evaluation
# Algorithm: eXtreme Gradient Boosting (XGBoost)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Force backend to prevent threading issues during plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings

warnings.filterwarnings('ignore')


# ================= Global Plot Settings (SCI-standard) =================
def set_scientific_style():
    plt.rcParams.update(plt.rcParamsDefault)
    FONT_FAMILY = "Times New Roman"
    FONT_SIZE = 16

    config = {
        "font.family": "serif",
        "font.serif": [FONT_FAMILY],
        "font.size": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.figsize": (8, 6),
        "figure.dpi": 300,
        "axes.linewidth": 2.0,
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "axes.labelsize": FONT_SIZE + 2,
        "axes.labelweight": "bold",
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.unicode_minus": False,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.fontsize": FONT_SIZE - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False
    }
    plt.rcParams.update(config)


# ================= Main Execution Block =================
if __name__ == "__main__":

    # Define relative paths for open-source repository
    INPUT_DATA_PATH = './data/training_cohort_final.xlsx'
    OUTPUT_DIR = './output/2_Model_Training_and_Evaluation/XGBoost/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Error: Dataset not found at {INPUT_DATA_PATH}. Please check the path.")
        exit()

    data = pd.read_excel(INPUT_DATA_PATH)

    # NOTE: Slicing starts from column 17 based on the specific dataset structure
    X = data.iloc[:, 17:]
    y = data.iloc[:, 0]

    # Handle potential infinite values gracefully
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate imbalance ratio for scale_pos_weight
    pos_count = sum(y == 1)
    neg_count = sum(y == 0)
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0

    # ================= 1. Hyperparameter Tuning =================
    print("Initializing XGBoost hyperparameter tuning via Grid Search...")

    # XGBoost natively handles missing values; pipeline imputation is optional
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric='auc',
        scale_pos_weight=imbalance_ratio,
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    model = grid_search.best_estimator_

    print(f"Optimization completed. Best cross-validated AUC: {best_score:.4f}")

    # ================= 2. Model Evaluation (5-Fold CV) =================
    print("Initiating 5-fold stratified cross-validation for comprehensive evaluation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tprs, fprs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    precisions, recalls = [], []
    base_recall = np.linspace(0, 1, 100)
    sum_conf_matrix = np.zeros((2, 2))
    results_list = []

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        sum_conf_matrix += cm
        tn, fp, fn, tp = cm.ravel()

        roc_val = roc_auc_score(y_test, y_pred_prob)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
        auc_pr_val = auc(recall_curve, precision_curve)

        results_list.append({
            'Fold': f'Fold {i + 1}',
            'AUC': roc_val,
            'AUC-PR': auc_pr_val,
            'Sensitivity (Recall)': sensitivity,
            'Specificity': specificity,
            'PPV (Precision)': ppv,
            'NPV': npv,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        })

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        precision_curve = precision_curve[::-1]
        recall_curve = recall_curve[::-1]
        interp_precision = np.interp(base_recall, recall_curve, precision_curve)
        interp_precision[0] = 1.0
        precisions.append(interp_precision)

    # ================= 3. Export Evaluation Metrics =================
    results_df = pd.DataFrame(results_list)
    mean_metrics = results_df.iloc[:, 1:].mean()
    mean_row = mean_metrics.to_dict()
    mean_row['Fold'] = 'Average'
    results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Best Value'])

    output_excel_path = os.path.join(OUTPUT_DIR, 'XGBoost_Evaluation_Metrics.xlsx')
    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Evaluation Metrics', index=False)
            params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
        print(f"Metrics and parameters successfully saved to {output_excel_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    # ================= 4. Data Visualization & PDF Export =================
    print("Generating visualizations (ROC, PR, Confusion Matrix)...")

    # --- ROC Curve ---
    set_scientific_style()
    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_roc_auc)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=.8, label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, 'XGBoost_ROC_Curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # --- PR Curve ---
    set_scientific_style()
    plt.figure()
    mean_precision = np.mean(precisions, axis=0)
    mean_auc_pr = auc(base_recall, mean_precision)
    std_precision = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)

    plt.plot(base_recall, mean_precision, color='g', label=r'Mean PR (AUC = %0.2f)' % mean_auc_pr)
    plt.fill_between(base_recall, precisions_lower, precisions_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left" if mean_auc_pr < 0.5 else "upper right")
    plt.savefig(os.path.join(OUTPUT_DIR, 'XGBoost_PR_Curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # --- Confusion Matrix ---
    set_scientific_style()
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.bottom'] = False
    plt.figure()
    sns.heatmap(sum_conf_matrix.astype(int), annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 18, "family": "Times New Roman"})
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=18)
    plt.ylabel('True Label', fontweight='bold', fontsize=18)
    plt.savefig(os.path.join(OUTPUT_DIR, 'XGBoost_Confusion_Matrix.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    print("Execution completed successfully.")