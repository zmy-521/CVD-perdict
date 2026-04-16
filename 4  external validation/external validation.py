# ==============================================================================
# External Validation Pipeline for Dual-Track Predictive Architecture
# Features: Dynamic Stratification, Calibration, DCA, and Cut-off Analysis
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Force backend to prevent threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve,
                             auc, precision_recall_curve, f1_score)
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings('ignore')

# ======================================================================
# 1. Path Configurations (Open-Source Relative Paths)
# ======================================================================
INPUT_DATA_PATH = "./data/external_validation_cohort.xlsx"

MODEL_DIR = "./models/"
PATH_PKL_GATEKEEPER = os.path.join(MODEL_DIR, "Model_Gatekeeper.pkl")
PATH_PKL_TRACK_A = os.path.join(MODEL_DIR, "Model_Track_A.pkl")
PATH_PKL_TRACK_B = os.path.join(MODEL_DIR, "Model_Track_B.pkl")
PATH_PKL_GLOBAL = os.path.join(MODEL_DIR, "Model_Global.pkl")

OUTPUT_MAIN_DIR = "./output/3_External_Validation/"
os.makedirs(OUTPUT_MAIN_DIR, exist_ok=True)


# ======================================================================
# 2. SCI-Standard Plot Settings (Times New Roman)
# ======================================================================
def set_scientific_style():
    plt.rcParams.update(plt.rcParamsDefault)
    FONT_FAMILY = "Times New Roman"
    FONT_SIZE = 14
    config = {
        "font.family": "serif", "font.serif": [FONT_FAMILY], "font.size": FONT_SIZE,
        "font.weight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.figsize": (7, 6), "figure.dpi": 300, "axes.linewidth": 2.0,
        "axes.labelsize": FONT_SIZE + 2, "axes.labelweight": "bold", "axes.titleweight": "bold",
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "xtick.direction": "in",
        "ytick.direction": "in", "axes.unicode_minus": False, "xtick.major.width": 2.0,
        "ytick.major.width": 2.0, "xtick.major.size": 6, "ytick.major.size": 6,
        "axes.spines.top": False, "axes.spines.right": False
    }
    plt.rcParams.update(config)


set_scientific_style()


def bold_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


# ======================================================================
# 3. Core Plotting & Evaluation Functions
# ======================================================================

def plot_clinical_cutoff_boxplot(y_true, y_prob, cutoff, out_dir):
    """Generates a strip-boxplot visualization for the clinical cut-off threshold."""
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_df = pd.DataFrame({
        'True Label': y_true.map({0: 'Negative', 1: 'Positive'}),
        'Predicted Probability': y_prob
    })
    palette = {'Negative': '#8DA0CB', 'Positive': '#FC8D62'}

    sns.boxplot(x='True Label', y='Predicted Probability', data=plot_df,
                palette=palette, width=0.5, boxprops=dict(alpha=0.4),
                showfliers=False, ax=ax, zorder=1)

    sns.stripplot(x='True Label', y='Predicted Probability', data=plot_df,
                  palette=palette, size=6, jitter=0.25, alpha=0.6, ax=ax, zorder=2)

    ax.axhline(y=cutoff, color='grey', linestyle='--', linewidth=2, zorder=3)
    ax.text(0.5, cutoff, f'Cutoff = {cutoff:.3f}', ha='center', va='bottom',
            fontsize=14, fontweight='bold', backgroundcolor='white', zorder=4)

    ax.annotate('', xy=(0.5, 1.0), xytext=(0.5, cutoff + 0.02),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(0.5, cutoff + (1.0 - cutoff) / 2, 'High Risk\n(Track A-ODKD)',
            ha='center', va='center', fontsize=12, fontweight='bold', backgroundcolor='white')

    ax.annotate('', xy=(0.5, 0.0), xytext=(0.5, cutoff - 0.02),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(0.5, cutoff / 2, 'Low Risk\n(Track B-Non-ODKD)',
            ha='center', va='center', fontsize=12, fontweight='bold', backgroundcolor='white')

    ax.set_xlabel('')
    ax.set_ylabel('Predicted Probability of ODKD')
    bold_ticks(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "0_Cutoff_StripBoxplot.pdf"), bbox_inches='tight')
    plt.close(fig)


def generate_cutoff_supplementary_table(y_true, y_prob, out_dir):
    """Generates a supplementary table evaluating multiple clinical thresholds."""
    thresholds_to_test = np.arange(0.10, 0.95, 0.05)
    table_rows = []

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    optimal_cutoff = roc_thresholds[best_idx]

    all_thresholds = sorted(list(thresholds_to_test) + [optimal_cutoff])

    for t in all_thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        auc_val = roc_auc_score(y_true, y_prob)

        row_label = f"{t:.3f} (Optimal)" if t == optimal_cutoff else f"{t:.3f}"

        table_rows.append({
            'Probability cutoff': row_label,
            'AUC': round(auc_val, 3),
            'Sensitivity': round(sens, 3),
            'Specificity': round(spec, 3),
            'PPV': round(ppv, 3),
            'NPV': round(npv, 3)
        })

    df_supp = pd.DataFrame(table_rows).drop_duplicates(subset=['Probability cutoff'])
    df_supp.to_excel(os.path.join(out_dir, "0_Supplementary_Table_Cutoff.xlsx"), index=False)


def plot_and_save(y_true, y_prob, model_name, out_dir):
    """Standard evaluation suite generating ROC, PR, CM, Calibration, and DCA."""
    os.makedirs(out_dir, exist_ok=True)

    # Enforced Navy Blue color scheme for professional SCI graphics
    UNIFIED_COLOR = "#004687"

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    y_pred_opt = (y_prob >= best_threshold).astype(int)

    roc_val = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_opt)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    brier = np.mean((y_prob - y_true) ** 2)

    # 1. ROC Curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color=UNIFIED_COLOR, lw=3.0, label=f'{model_name}\n(AUC = {roc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='#B0B0B0', lw=2, linestyle='--')
    ax.set_xlim([-0.02, 1.02]);
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate');
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right", frameon=False, prop={'weight': 'bold', 'family': 'Times New Roman'})
    bold_ticks(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_ROC.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 2. PR Curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color=UNIFIED_COLOR, lw=3.0, label=f'{model_name}\n(PR-AUC = {pr_auc:.3f})')
    ax.set_xlim([-0.02, 1.02]);
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall');
    ax.set_ylabel('Precision')
    ax.legend(loc="lower left", frameon=False, prop={'weight': 'bold', 'family': 'Times New Roman'})
    bold_ticks(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_PR.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 3. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.rcParams['axes.spines.top'] = True;
    plt.rcParams['axes.spines.right'] = True
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                annot_kws={"size": 18, "weight": "bold", "family": "Times New Roman"}, ax=ax)
    ax.set_xlabel('Predicted Label');
    ax.set_ylabel('True Label')
    bold_ticks(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_CM.pdf"), bbox_inches='tight')
    plt.close(fig)
    plt.rcParams['axes.spines.top'] = False;
    plt.rcParams['axes.spines.right'] = False

    # 4. Calibration Curve
    fop, mpv = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color='#B0B0B0', linestyle="--", lw=2, label="Perfect Calibration")
    ax.plot(mpv, fop, "s-", color=UNIFIED_COLOR, lw=2.5, markersize=8, label=f'Brier = {brier:.3f}')
    ax.set_ylabel("Fraction of Positives");
    ax.set_xlabel("Mean Predicted Probability")
    ax.legend(loc="lower right", frameon=False, prop={'weight': 'bold', 'family': 'Times New Roman'})
    bold_ticks(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "4_Calibration.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 5. Decision Curve Analysis (DCA)
    thresh_dca = np.arange(0.01, 0.81, 0.01)
    n = len(y_true)
    nb_model, nb_all = [], []
    for t in thresh_dca:
        yp = (y_prob >= t).astype(int)
        _, fp_l, _, tp_l = confusion_matrix(y_true, yp).ravel()
        nb_model.append((tp_l / n) - (fp_l / n) * (t / (1 - t)))
        nb_all.append((np.sum(y_true == 1) / n) - (np.sum(y_true == 0) / n) * (t / (1 - t)))
    fig, ax = plt.subplots()
    ax.plot(thresh_dca, nb_model, color=UNIFIED_COLOR, lw=3.0, label=f'{model_name}')
    ax.plot(thresh_dca, nb_all, color='#808080', lw=2, linestyle='--', label='Treat All')
    ax.plot([0, 1], [0, 0], color='black', lw=2, linestyle='-', label='Treat None')
    ax.set_xlim([0, 0.8]);
    ax.set_ylim([-0.05, max(max(nb_model), max(nb_all)) + 0.05])
    ax.set_xlabel('Threshold Probability');
    ax.set_ylabel('Net Benefit')
    ax.legend(loc="upper right", frameon=False, prop={'weight': 'bold', 'family': 'Times New Roman'})
    bold_ticks(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "5_DCA.pdf"), bbox_inches='tight')
    plt.close(fig)

    return {
        'Model Phase': model_name, 'AUC': round(roc_val, 3), 'Accuracy': round(acc, 3),
        'Sensitivity': round(sensitivity, 3), 'Specificity': round(specificity, 3),
        'PPV': round(ppv, 3), 'NPV': round(npv, 3), 'F1-Score': round(f1, 3),
        'PR-AUC': round(pr_auc, 3), 'Brier Score': round(brier, 3),
        'Cutoff': round(best_threshold, 3), 'Cohort Size': len(y_true)
    }


# ======================================================================
# 4. Main Execution: Dynamic Stratification Pipeline
# ======================================================================
if __name__ == "__main__":
    print("-" * 60)
    print("🚀 Initiating Dual-Track Dynamic Validation Pipeline...")

    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Error: Validation dataset not found at {INPUT_DATA_PATH}")
        exit()

    df_val = pd.read_excel(INPUT_DATA_PATH)
    results_list = []

    # --- Phase 1: ODKD Gatekeeper ---
    print("\n[1/4] Executing ODKD Gatekeeper (Evaluating total cohort)...")
    gatekeeper = load(PATH_PKL_GATEKEEPER)
    features_gk = ['Age', 'BUN', 'RDW', 'SUA', 'HbA1c', 'Cl', 'A/G', 'NEU#']
    y_true_gk = df_val['ODKD-Label']
    y_prob_gk = gatekeeper.predict_proba(df_val[features_gk])[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true_gk, y_prob_gk)
    best_cutoff_gk = thresholds[np.argmax(tpr - fpr)]

    df_val['Machine_Predicted_ODKD'] = (y_prob_gk >= best_cutoff_gk).astype(int)
    df_track_A = df_val[df_val['Machine_Predicted_ODKD'] == 1].copy()
    df_track_B = df_val[df_val['Machine_Predicted_ODKD'] == 0].copy()

    print(f"   => Optimal Gatekeeper Cutoff = {best_cutoff_gk:.4f}")
    print(f"   => [Stratification Complete] Track A: {len(df_track_A)} | Track B: {len(df_track_B)}")

    gk_out_dir = os.path.join(OUTPUT_MAIN_DIR, "1_ODKD_Gatekeeper")
    os.makedirs(gk_out_dir, exist_ok=True)

    plot_clinical_cutoff_boxplot(y_true_gk, y_prob_gk, best_cutoff_gk, gk_out_dir)
    generate_cutoff_supplementary_table(y_true_gk, y_prob_gk, gk_out_dir)
    res_gk = plot_and_save(y_true_gk, y_prob_gk, "ODKD Gatekeeper", gk_out_dir)
    results_list.append(res_gk)

    # --- Phase 2: Track A-ODKD ---
    if len(df_track_A) > 0:
        print("\n[2/4] Executing Track A-ODKD (Evaluating predicted positive cohort)...")
        model_track_A = load(PATH_PKL_TRACK_A)
        features_A = ['ALT', 'Age', 'RDW', 'Non-HDL-C', 'PLT', 'HbA1c', 'Cl', 'SCr']
        y_true_A = df_track_A['CVD-Label']
        y_prob_A = model_track_A.predict_proba(df_track_A[features_A])[:, 1]
        df_track_A['Predicted_CVD_Prob'] = y_prob_A
        res_A = plot_and_save(y_true_A, y_prob_A, "Track A-ODKD", os.path.join(OUTPUT_MAIN_DIR, "2_Track_A_ODKD"))
        results_list.append(res_A)

    # --- Phase 3: Track B-Non-ODKD ---
    if len(df_track_B) > 0:
        print("\n[3/4] Executing Track B-Non-ODKD (Evaluating predicted negative cohort)...")
        model_track_B = load(PATH_PKL_TRACK_B)
        features_B = ['SUA', 'Age', 'K', 'RDW', 'Non-HDL-C', 'MCV', 'SCr']
        y_true_B = df_track_B['CVD-Label']
        y_prob_B = model_track_B.predict_proba(df_track_B[features_B])[:, 1]
        df_track_B['Predicted_CVD_Prob'] = y_prob_B
        res_B = plot_and_save(y_true_B, y_prob_B, "Track B-Non-ODKD",
                              os.path.join(OUTPUT_MAIN_DIR, "3_Track_B_Non_ODKD"))
        results_list.append(res_B)

    # --- Phase 4: Global Model (Baseline) ---
    print("\n[4/4] Executing Global Model (Baseline) (Evaluating total cohort for comparison)...")
    model_global = load(PATH_PKL_GLOBAL)
    features_global = ['SUA', 'ALT', 'MON#', 'Age', 'LYM#', 'RDW', 'Non-HDL-C', 'K', 'PLT', 'MCV', 'SCr']
    y_true_global = df_val['CVD-Label']
    y_prob_global = model_global.predict_proba(df_val[features_global])[:, 1]
    res_global = plot_and_save(y_true_global, y_prob_global, "Global Model (Baseline)",
                               os.path.join(OUTPUT_MAIN_DIR, "4_Global_Baseline"))
    results_list.append(res_global)

    # ======================================================================
    # 5. Export Final Summary Tables
    # ======================================================================
    df_summary = pd.DataFrame(results_list)
    df_summary.to_excel(os.path.join(OUTPUT_MAIN_DIR, "Table_1_Dual_Track_Validation_Summary.xlsx"), index=False)

    df_val_final = pd.concat([df_track_A, df_track_B]).sort_index()
    df_val_final['Global_CVD_Prob'] = y_prob_global
    df_val_final.to_excel(os.path.join(OUTPUT_MAIN_DIR, "Data_Aligned_Probabilities_Summary.xlsx"), index=False)

    print("-" * 60)
    print("🎉 Pipeline successfully executed! All curves unified to Navy Blue (#004687). Check the output directory.")