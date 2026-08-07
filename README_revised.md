# 🫀 Phenotype-Aware Dual-Track CVD Stratification System

A Streamlit research prototype accompanying the manuscript **“Development and Validation of a Dual-Track Machine Learning System for Cardiovascular Risk Reclassification: A Gatekeeper-Guided Approach.”**

The application compares a conventional unstratified global model with a gatekeeper-guided dual-track framework for cross-sectional cardiovascular disease (CVD) stratification in adults with type 2 diabetes. The gatekeeper estimates the probability of occult diabetic kidney disease (ODKD) using routinely available demographic and blood-based variables and routes patients to phenotype-specific downstream models without requiring urinary measurements as model inputs.

> **Research context:** The application is intended to demonstrate model-estimated probabilities, gatekeeper-based routing, probability reassignment, and individualized SHAP explanations. It is not intended to establish an ODKD diagnosis, predict future cardiovascular events, or guide treatment decisions.

---

## ✨ Key Features

- **Gatekeeper-guided routing:** Estimates ODKD probability and routes patients using the primary gatekeeper cutoff of **0.532** (Track A if probability ≥0.532; Track B if probability <0.532).
- **Phenotype-aware CVD models:** Applies separate downstream models after routing and compares their outputs with a conventional global model.
- **Head-to-head comparison:** Displays model-estimated probabilities from the dual-track framework and the global model for the same patient.
- **Probability-category display:** Uses the web-display categories **<30%, 30% to <50%, and ≥50%** to visualize probability reassignment. These categories are display categories and **not treatment or intervention thresholds**.
- **SHAP interpretability:** Provides individualized feature-attribution plots to show how input variables contribute to each model output.
- **Routine-variable input:** Uses routinely available demographic and blood-based variables; urinary measurements are not required as gatekeeper inputs.

---

## 📋 Clinical Input Variables

The web application uses routinely available demographic and laboratory variables required by the gatekeeper and downstream/global models.

| Category | Variable | Description | Type / Unit |
| :--- | :--- | :--- | :--- |
| **Basic/Metabolic** | Age | Age | Continuous (years) |
| | HbA1c | Glycated hemoglobin | Continuous (%) |
| | SUA | Serum uric acid | Continuous (μmol/L) |
| | Non-HDL-C | Non-high-density lipoprotein cholesterol | Continuous (mmol/L) |
| **CBC** | RDW | Red cell distribution width | Continuous (%) |
| | NEU# | Neutrophil count | Continuous (10^9/L) |
| | LYM# | Lymphocyte count | Continuous (10^9/L) |
| | MON# | Monocyte count | Continuous (10^9/L) |
| | PLT | Platelet count | Continuous (10^9/L) |
| | MCV | Mean corpuscular volume | Continuous (fL) |
| **Organ & Ions** | BUN | Blood urea nitrogen | Continuous (mmol/L) |
| | SCr | Serum creatinine | Continuous (μmol/L) |
| | ALT | Alanine aminotransferase | Continuous (U/L) |
| | A/G Ratio | Albumin-to-globulin ratio | Continuous |
| | Cl | Chloride | Continuous (mmol/L) |
| | K | Potassium | Continuous (mmol/L) |

---

## 🚀 Deployment

### Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/zmy-521/CVD-predict.git
cd CVD-predict

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

### Streamlit Cloud

1. Upload this repository to your GitHub account.
2. Go to Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy the application.

---

## 📁 File Structure

```text
CVD-predict/
├── app.py                 # Streamlit application
├── Model_Gatekeeper.pkl   # Gatekeeper model
├── Model_Track_A.pkl      # Track A downstream model
├── Model_Track_B.pkl      # Track B downstream model
├── Model_Global.pkl       # Conventional global model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🧬 System Architecture

- **Gatekeeper task:** Estimate the probability of the ODKD phenotype from demographic and blood-based variables.
- **Primary routing cutoff:** **0.532**, selected by maximizing the Youden index in the external validation cohort.
- **Sensitivity-analysis cutoff:** **0.489**, derived from out-of-fold predictions in the NHANES derivation cohort.
- **Routing rule:** Gatekeeper probability ≥0.532 → **Track A**; probability <0.532 → **Track B**.
- **Downstream endpoint:** Model-estimated probability of **prevalent or concurrent CVD**, not incident future CVD.
- **Web-display categories:** **<30%, 30% to <50%, and ≥50%**. These are visualization categories and should not be interpreted as clinically actionable treatment thresholds.
- **Clinical role of the gatekeeper:** The gatekeeper is a phenotype-informed routing and screening-prioritization tool. It does not replace UACR-based diagnosis of ODKD.

---

## ⚠️ Disclaimer

This application is provided for **research, peer-review, and educational purposes only**. It has not been prospectively validated for clinical deployment and should not be used to diagnose ODKD, predict future cardiovascular events, determine treatment eligibility, or replace clinician judgment. Local recalibration and prospective evaluation are required before any clinical use.
