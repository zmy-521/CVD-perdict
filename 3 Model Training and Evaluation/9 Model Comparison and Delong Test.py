# ==============================================================================
# Comprehensive Model Comparison and Statistical Evaluation
# Feature: 5-Fold CV, Combined ROC/PR Curves, and DeLong Test for AUC Comparison
# ==============================================================================

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib

matplotlib.use('Agg')  # Force backend to prevent threading issues during plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve

# --- Import all 9 machine learning algorithms ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
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
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.unicode_minus": False,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.fontsize": FONT_SIZE - 4,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False
    }
    plt.rcParams.update(config)


# ================= Main Execution Block =================
if __name__ == "__main__":

    # Define relative paths for open-source repository
    INPUT_DATA_PATH = './data/training_cohort_selected_features.xlsx'  # Adjust file name if needed
    OUTPUT_DIR = './output/2_Model_Training_and_Evaluation/Model_Comparison/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_metrics_path = os.path.join(OUTPUT_DIR, 'Comprehensive_Metrics_Comparison.xlsx')
    output_delong_path = os.path.join(OUTPUT_DIR, 'DeLong_Test_Results.xlsx')

    print("Loading data...")
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Error: Dataset not found at {INPUT_DATA_PATH}. Please check the path.")
        exit()

    data = pd.read_excel(INPUT_DATA_PATH)

    # NOTE: Adjust slicing index based on your specific dataset
    X = data.iloc[:, 17:]
    y = data.iloc[:, 0]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    imbalance_ratio = sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1.0

    # ============================================================
    # 1. Initialize the 9 Models with Optimal Pipelines
    # ============================================================
    print("Initializing 9 optimized machine learning models...")

    models = {
        'LR': Pipeline([
            ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.01, penalty='l1', solver='saga', class_weight='balanced', random_state=42,
                                      max_iter=2000))
        ]),
        'KNN': Pipeline([
            ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance', p=1))
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
            ('svm', SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42))
        ]),
        'ANN': Pipeline([
            ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
            ('ann', MLPClassifier(hidden_layer_sizes=(16, 8), activation='tanh', alpha=0.001, learning_rate_init=0.001,
                                  max_iter=1000, early_stopping=True, random_state=42))
        ]),
        'RF': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=2,
                                          class_weight='balanced', random_state=42, n_jobs=-1))
        ]),
        'AdaBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('ada', AdaBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('xgb',
             XGBClassifier(n_estimators=300, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=imbalance_ratio, eval_metric='auc', random_state=42, n_jobs=-1))
        ]),
        'LightGBM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('lgbm', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.01, num_leaves=31, max_depth=4, subsample=0.8,
                                        colsample_bytree=0.8, objective='binary', metric='auc', class_weight='balanced',
                                        random_state=42, n_jobs=-1, verbose=-1))
        ]),
        'DT': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('dt', DecisionTreeClassifier(criterion='gini', max_depth=3, max_features=None, min_samples_leaf=1,
                                          min_samples_split=2, class_weight='balanced', random_state=42))
        ])
    }

    # ============================================================
    # 2. Execute 5-Fold CV & Collect Out-Of-Fold (OOF) Predictions
    # ============================================================
    print("Executing 5-Fold Cross Validation for all models...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    plot_data = {name: {'tprs': [], 'precisions': [], 'aucs': [], 'auc_prs': []} for name in models.keys()}
    metrics_data = {name: {'AUC': [], 'AUC-PR': [], 'Sensitivity': [], 'Specificity': [],
                           'PPV': [], 'NPV': [], 'Accuracy': [], 'F1': [],
                           'TP_sum': 0, 'TN_sum': 0, 'FP_sum': 0, 'FN_sum': 0} for name in models.keys()}

    # OOF Prediction Collector for DeLong Test
    n_samples = len(y)
    oof_preds = {name: np.zeros(n_samples) for name in models.keys()}
    y_true_ordered = np.zeros(n_samples)

    mean_fpr = np.linspace(0, 1, 100)
    base_recall = np.linspace(0, 1, 100)

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"  Processing Fold {i + 1}/5...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        y_true_ordered[test_index] = y_test

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            oof_preds[name][test_index] = y_pred_prob

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics_data[name]['TP_sum'] += tp
            metrics_data[name]['TN_sum'] += tn
            metrics_data[name]['FP_sum'] += fp
            metrics_data[name]['FN_sum'] += fn

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

            roc_val = roc_auc_score(y_test, y_pred_prob)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
            auc_pr_val = auc(recall_curve, precision_curve)

            metrics_data[name]['AUC'].append(roc_val)
            metrics_data[name]['AUC-PR'].append(auc_pr_val)
            metrics_data[name]['Sensitivity'].append(sensitivity)
            metrics_data[name]['Specificity'].append(specificity)
            metrics_data[name]['PPV'].append(ppv)
            metrics_data[name]['NPV'].append(npv)
            metrics_data[name]['Accuracy'].append(accuracy)
            metrics_data[name]['F1'].append(f1)

            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            plot_data[name]['tprs'].append(interp_tpr)
            plot_data[name]['aucs'].append(roc_val)
            plot_data[name]['auc_prs'].append(auc_pr_val)

            precision_curve = precision_curve[::-1]
            recall_curve = recall_curve[::-1]
            interp_precision = np.interp(base_recall, recall_curve, precision_curve)
            interp_precision[0] = 1.0
            plot_data[name]['precisions'].append(interp_precision)

    # ============================================================
    # 3. Export Comprehensive Summary to Excel
    # ============================================================
    print("\nExporting comprehensive metrics summary...")
    final_summary = []
    for name in models.keys():
        model_mean_metrics = {
            'Model': name,
            'AUC': np.mean(metrics_data[name]['AUC']),
            'AUC-PR': np.mean(metrics_data[name]['AUC-PR']),
            'Sensitivity': np.mean(metrics_data[name]['Sensitivity']),
            'Specificity': np.mean(metrics_data[name]['Specificity']),
            'PPV': np.mean(metrics_data[name]['PPV']),
            'NPV': np.mean(metrics_data[name]['NPV']),
            'Accuracy': np.mean(metrics_data[name]['Accuracy']),
            'F1 Score': np.mean(metrics_data[name]['F1']),
            'Total TP': metrics_data[name]['TP_sum'], 'Total TN': metrics_data[name]['TN_sum'],
            'Total FP': metrics_data[name]['FP_sum'], 'Total FN': metrics_data[name]['FN_sum']
        }
        final_summary.append(model_mean_metrics)

    summary_df = pd.DataFrame(final_summary).sort_values(by='AUC', ascending=False).reset_index(drop=True)

    try:
        summary_df.to_excel(output_metrics_path, index=False, float_format="%.4f")
        print(f"Success: Comprehensive metrics saved to {output_metrics_path}")
    except Exception as e:
        print(f"Error saving summary Excel file:\n{e}")

    # ============================================================
    # 4. DeLong Test (Statistical Comparison of ROC AUC)
    # ============================================================
    print("\nExecuting DeLong Test for Statistical Significance...")


    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T
        return T2


    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]
        tx = np.empty([k, m], dtype=np.float64)
        ty = np.empty([k, n], dtype=np.float64)
        tz = np.empty([k, m + n], dtype=np.float64)
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov


    def calc_p_value(y_true, y_pred_A, y_pred_B):
        y_true = np.array(y_true).reshape(-1)
        y_pred_A = np.array(y_pred_A).reshape(-1)
        y_pred_B = np.array(y_pred_B).reshape(-1)
        order = np.argsort(y_true)[::-1]
        y_true = y_true[order]
        y_pred_A = y_pred_A[order]
        y_pred_B = y_pred_B[order]
        predictions = np.vstack((y_pred_A, y_pred_B))
        label_1_count = int(np.sum(y_true))
        aucs, delongcov = fastDeLong(predictions, label_1_count)
        l = np.array([1, -1])
        sigma_diff = np.sqrt(np.dot(np.dot(l, delongcov), l.T))
        z_score = 0 if sigma_diff == 0 else (aucs[0] - aucs[1]) / sigma_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        return aucs[0], aucs[1], p_value


    # Identify the best model based on overall OOF AUC
    model_aucs = {name: roc_auc_score(y_true_ordered, preds) for name, preds in oof_preds.items()}
    best_model_name = max(model_aucs, key=model_aucs.get)
    print(f"Champion Model Identified: {best_model_name} (AUC = {model_aucs[best_model_name]:.4f})")

    comparison_results = []
    for name in models.keys():
        if name == best_model_name: continue
        auc_best, auc_other, p_val = calc_p_value(y_true_ordered, oof_preds[best_model_name], oof_preds[name])
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        comparison_results.append({
            'Champion Model (A)': best_model_name,
            'Comparator Model (B)': name,
            'AUC A': auc_best,
            'AUC B': auc_other,
            'Diff (A-B)': auc_best - auc_other,
            'P-value': p_val,
            'Significance': significance
        })

    df_delong = pd.DataFrame(comparison_results).sort_values(by='P-value').reset_index(drop=True)
    try:
        df_delong.to_excel(output_delong_path, index=False)
        print(f"Success: DeLong Test results saved to {output_delong_path}")
    except Exception as e:
        print(f"Error saving DeLong Excel file:\n{e}")

    # ============================================================
    # 5. Generate Combined ROC & PR Curves (PDF)
    # ============================================================
    print("\nGenerating combined ROC and PR curve PDFs...")

    color_map = {
        'LR': '#9467bd', 'SVM': '#1f77b4', 'ANN': '#ff7f0e', 'KNN': '#d62728',
        'RF': '#17becf', 'AdaBoost': '#8c564b', 'XGBoost': '#2ca02c',
        'LightGBM': '#bcbd22', 'DT': '#7f7f7f'
    }

    # --- Plot 1: Combined ROC ---
    set_scientific_style()
    plt.figure()
    for name in summary_df['Model']:
        mean_tpr = np.mean(plot_data[name]['tprs'], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(plot_data[name]['aucs'])
        plt.plot(mean_fpr, mean_tpr, color=color_map[name], lw=2.0, alpha=0.85, zorder=1,
                 label=f'{name} (AUC = {mean_auc:.3f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2.0, color='gray', alpha=0.8, label='Chance')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, '9Models_Combined_ROC.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # --- Plot 2: Combined PR ---
    set_scientific_style()
    plt.figure()
    for name in summary_df['Model']:
        mean_precision = np.mean(plot_data[name]['precisions'], axis=0)
        mean_auc_pr = np.mean(plot_data[name]['auc_prs'])
        plt.plot(base_recall, mean_precision, color=color_map[name], lw=2.0, alpha=0.85, zorder=1,
                 label=f'{name} (AUC-PR = {mean_auc_pr:.3f})')

    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(OUTPUT_DIR, '9Models_Combined_PR.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    print("\nExecution completely successful. Combined PDFs and Metrics generated!")