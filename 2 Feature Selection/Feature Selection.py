# ==============================================================================
# Feature Selection Pipeline for Dual-Track Predictive Modeling
# Methods: Univariate Logistic Regression, LASSO Regression, and Boruta Algorithm
# ==============================================================================

import matplotlib

matplotlib.use('Agg')  # Force backend to prevent threading issues during plotting

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from matplotlib_venn import venn3
import warnings

warnings.filterwarnings("ignore")

# ================= Global Plot Settings (SCI-standard) =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# ================= Core Feature Selection Function =================
def run_feature_selection(df_subset, target, exclude_cols, output_dir, scenario_name):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Executing Feature Selection For: {scenario_name}")
    print(f"Output Directory: {output_dir}")
    print(f"{'=' * 60}")

    # 1. Data Preparation
    features = [c for c in df_subset.columns if c not in exclude_cols and c != target]
    X = df_subset[features]
    y = df_subset[target]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # 2. Univariate Logistic Regression
    print(f"  [1/4] Running Univariate Logistic Regression...")
    univ_selected, univ_results = [], []
    for col in features:
        X_univ = sm.add_constant(X[col])
        try:
            model = sm.Logit(y, X_univ).fit(disp=0)
            coef, se, p_val = model.params[col], model.bse[col], model.pvalues[col]
            if p_val < 0.05:
                univ_selected.append(col)

            or_val = np.exp(coef)
            ci_lower, ci_upper = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
            univ_results.append({
                'Variable': col,
                'OR': round(or_val, 3),
                '95% CI': f"{round(ci_lower, 3)}-{round(ci_upper, 3)}",
                'P value': round(p_val, 3) if p_val >= 0.001 else "<0.001"
            })
        except:
            pass
    pd.DataFrame(univ_results).to_excel(os.path.join(output_dir, "Supplement_Table_Univariate.xlsx"), index=False)

    # 3. LASSO Regression
    print(f"  [2/4] Running LASSO Regression & Path Visualization...")
    Cs_grid = np.logspace(-3, 3, 100)
    lasso_cv = LogisticRegressionCV(Cs=Cs_grid, cv=10, penalty='l1', solver='liblinear', random_state=42,
                                    scoring='neg_log_loss')
    lasso_cv.fit(X_scaled, y)

    lasso_coefs_final = pd.Series(lasso_cv.coef_[0], index=features)
    lasso_selected = lasso_coefs_final[lasso_coefs_final != 0].index.tolist()

    lambdas, log_lambdas = 1 / lasso_cv.Cs_, np.log10(1 / lasso_cv.Cs_)
    mean_scores, std_scores = -lasso_cv.scores_[1].mean(axis=0), lasso_cv.scores_[1].std(axis=0)

    min_idx = np.argmin(mean_scores)
    lambda_min = log_lambdas[min_idx]
    threshold = mean_scores[min_idx] + std_scores[min_idx]
    idx_1se_list = np.where((mean_scores <= threshold) & (log_lambdas >= lambda_min))[0]
    lambda_1se = log_lambdas[idx_1se_list[-1]] if len(idx_1se_list) > 0 else lambda_min

    coefs = lasso_cv.coefs_paths_[1].mean(axis=0)
    l1_norms = np.abs(coefs).sum(axis=1)
    sort_idx_l1 = np.argsort(l1_norms)

    # Plot B: LASSO Coefficient Path
    plt.figure(figsize=(8, 6))
    plt.plot(l1_norms[sort_idx_l1], coefs[sort_idx_l1, :])
    plt.xlabel('L1 Norm')
    plt.ylabel('Coefficients')
    plt.ylim(-0.6, 0.6)
    plt.title('B   LASSO Coefficient Profile Plot', loc='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_B_LASSO_Path.pdf"))
    plt.close()

    # Plot C: Cross-Validation Curve
    plt.figure(figsize=(8, 6))
    plot_mask = (log_lambdas >= lambda_min - 2.0) & (log_lambdas <= lambda_min + 0.6)
    pll, pms, pss = log_lambdas[plot_mask], mean_scores[plot_mask], std_scores[plot_mask]
    sort_idx = np.argsort(pll)
    plt.errorbar(pll[sort_idx], pms[sort_idx], yerr=pss[sort_idx], fmt='o', color='red', ecolor='gray', elinewidth=1,
                 capsize=3)
    plt.axvline(x=lambda_min, color='black', linestyle='--', label='lambda.min')
    plt.axvline(x=lambda_1se, color='black', linestyle=':', label='lambda.1se')
    plt.xlabel('Log(λ)')
    plt.ylabel('Binomial Deviance (Log Loss)')
    plt.title('C   10-fold CV for Tuning Parameter', loc='left')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_C_LASSO_CV.pdf"))
    plt.close()

    # 4. Boruta Algorithm
    print(f"  [3/4] Running Boruta Algorithm (This may take a while)...")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
    boruta.fit(X_scaled.values, y.values)
    boruta_selected = X.columns[boruta.support_].tolist()

    rf_eval = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)
    rf_eval.fit(X_scaled, y)
    raw_importances = np.array([tree.feature_importances_ for tree in rf_eval.estimators_])
    tree_importances = raw_importances.reshape(50, 10, -1).mean(axis=1)

    shadow_mask = ~boruta.support_
    if shadow_mask.sum() > 0:
        z_importances = (tree_importances - np.mean(tree_importances[:, shadow_mask])) / (
                    np.std(tree_importances[:, shadow_mask]) + 1e-6)
    else:
        z_importances = tree_importances

    df_box = pd.DataFrame(z_importances, columns=features)
    df_box_melt = df_box.melt(var_name='Feature', value_name='Importance (Z-score)')
    status_dict = {f: ('Confirmed' if boruta.support_[i] else 'Tentative' if boruta.support_weak_[i] else 'Rejected')
                   for i, f in enumerate(features)}
    df_box_melt['Status'] = df_box_melt['Feature'].map(status_dict)

    order = df_box.median().sort_values(ascending=True).index.tolist()
    shadow_max_z = df_box[df_box.columns[shadow_mask]].median().max() if shadow_mask.sum() > 0 else 0

    # Plot A: Boruta Feature Importance Boxplot
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Feature', y='Importance (Z-score)', hue='Status', dodge=False, data=df_box_melt, order=order,
                palette={'Confirmed': '#2ca02c', 'Tentative': '#ff7f0e', 'Rejected': '#d62728'}, showfliers=False)
    plt.legend([], [], frameon=False)
    plt.axhline(y=shadow_max_z, color='blue', linestyle='--', label='Shadow Max')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Importance (Z-Score)')
    plt.title('A   Boruta Feature Selection', loc='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_A_Boruta_Boxplot.pdf"))
    plt.close()

    # 5. Venn Diagram & Final Dataset Export
    print(f"  [4/4] Plotting Venn Diagram and generating optimal dataset...")
    set_univ, set_lasso, set_boruta = set(univ_selected), set(lasso_selected), set(boruta_selected)
    ultimate_features = list(set_univ & set_lasso & set_boruta)

    plt.figure(figsize=(8, 8))
    venn = venn3([set_univ, set_lasso, set_boruta],
                 set_labels=('Univariate\n(P<0.05)', 'LASSO\n(Selected)', 'Boruta\n(Confirmed)'))
    for text in venn.set_labels:
        if text: text.set_fontsize(16)
    for text in venn.subset_labels:
        if text: text.set_fontsize(22)
    plt.title('D   Intersection of Core Predictive Features', loc='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_D_Venn_Diagram.pdf"))
    plt.close()

    # Generate Summary Document
    text_content = f"""==================================================
Feature Selection Summary: {scenario_name}
==================================================

1. Univariate Logistic Regression ({len(univ_selected)} variables):
{', '.join(univ_selected)}

2. LASSO Regression ({len(lasso_selected)} variables):
{', '.join(lasso_selected)}

3. Boruta Algorithm Confirmed ({len(boruta_selected)} variables):
{', '.join(boruta_selected)}

--------------------------------------------------
Ultimate Intersection ({len(ultimate_features)} variables)
(Features to be included in downstream machine learning modeling):
{', '.join(ultimate_features)}
==================================================
"""
    with open(os.path.join(output_dir, "Feature_Selection_Summary.txt"), "w", encoding='utf-8') as f:
        f.write(text_content)

    # Reorder columns: keep non-features on the left, ultimate features on the right
    non_feature_cols = [c for c in df_subset.columns if c not in features]
    final_export_cols = non_feature_cols + ultimate_features

    # Export finalized dataset
    excel_filename = f"Dataset_{scenario_name}_{len(ultimate_features)}_Features.xlsx"
    df_subset[final_export_cols].to_excel(os.path.join(output_dir, excel_filename), index=False)

    print(f"Task Completed! Extracted {len(ultimate_features)} core features. Saved as: {excel_filename}\n")


# ================= Main Execution Block =================
if __name__ == "__main__":

    # 🚨 NOTE: Ensure your dataset is placed in the './data/' folder relative to this script
    input_path = "./data/training_cohort_final.xlsx"

    if not os.path.exists(input_path):
        print(f"Error: Dataset not found at {input_path}. Please check the file path.")
    else:
        df_main = pd.read_excel(input_path)

        # Baseline clinical variables to exclude from feature selection logic
        base_exclude = ['Patient_ID', 'UACR', 'UAlb', 'UCr', 'eGFR',
                        'Heart_Failure', 'CHD', 'Angina', 'Heart_Attack', 'Stroke',
                        'Hypertension', 'High_Cholesterol', 'Cancer', 'Kidney_Failure', 'PIR', 'Weight_kg']

        # Task 1: Gatekeeper Model (Predicting ODKD in total cohort)
        run_feature_selection(
            df_subset=df_main,
            target='ODKD_Label',
            exclude_cols=base_exclude + ['CVD_Label'],
            output_dir="./output/1_Global_ODKD_Gatekeeper",
            scenario_name="Global_ODKD_Gatekeeper"
        )

        # Task 2: Track A Model (Predicting CVD in ODKD-positive cohort)
        run_feature_selection(
            df_subset=df_main[df_main['ODKD_Label'] == 1],
            target='CVD_Label',
            exclude_cols=base_exclude + ['ODKD_Label'],
            output_dir="./output/2_Track_A_ODKD_Positive",
            scenario_name="Track_A_ODKD_Positive"
        )

        # Task 3: Track B Model (Predicting CVD in ODKD-negative cohort)
        run_feature_selection(
            df_subset=df_main[df_main['ODKD_Label'] == 0],
            target='CVD_Label',
            exclude_cols=base_exclude + ['ODKD_Label'],
            output_dir="./output/3_Track_B_ODKD_Negative",
            scenario_name="Track_B_ODKD_Negative"
        )

        # Task 4: Global Baseline Model (Predicting CVD in total cohort without stratification)
        run_feature_selection(
            df_subset=df_main,
            target='CVD_Label',
            exclude_cols=base_exclude + ['ODKD_Label'],
            output_dir="./output/4_Global_CVD_Baseline",
            scenario_name="Global_CVD_Baseline"
        )

        print("All feature selection tasks successfully executed and datasets exported.")