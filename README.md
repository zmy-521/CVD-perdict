# 🫀 Precision CVD Risk Stratification: Heterogeneity-Aware Dual-Track System

A Streamlit web application demonstrating a novel **Dual-Track Triage System** for cardiovascular disease (CVD) risk prediction. This tool provides head-to-head comparisons between a conventional unstratified global model and our proposed heterogeneity-aware mechanism guided by Occult Diabetic Kidney Disease (ODKD) status.

> **Research Context:** This application serves as the interactive clinical demonstration for evaluating risk reclassification benefits—specifically Net Reclassification Improvement (NRI) avoiding over-treatment, and Integrated Discrimination Improvement (IDI) rescuing missed risks.

---

## ✨ Key Features

- **Gatekeeper Triage Mechanism:** Automatically routes patients to specialized risk tracks (Track A for ODKD+, Track B for Non-ODKD).
- **Head-to-Head Comparison:** Real-time benchmarking against a standard unstratified Global Model.
- **Dynamic Reclassification Alerts:** Automatically detects and highlights precision downgrading (NRI) and upgrading (IDI) events crossing the 50% clinical cutoff.
- **Academic-Grade UI:** High-fidelity "vernier caliper" style risk bars with distinct clinical thresholds (<30% Low, 30-50% Intermediate, ≥50% High).
- **Comprehensive Variable Input:** Utilizes 16 routinely accessible clinlabomics features.

---

## 📋 Clinical Input Variables

All features are designed to be easily obtainable from routine blood tests and basic metabolic panels in primary care settings.

| Category | Variable | Description | Type / Unit |
| :--- | :--- | :--- | :--- |
| **Basic/Metabolic** | Age | Patient Age | Continuous (years) |
| | HbA1c | Glycated Hemoglobin | Continuous (%) |
| | SUA | Serum Uric Acid | Continuous (μmol/L) |
| | Non-HDL-C | Non-High-Density Lipoprotein Cholesterol | Continuous (mmol/L) |
| **CBC** | RDW | Red Cell Distribution Width | Continuous (%) |
| | NEU# | Neutrophil Count | Continuous (10^9/L) |
| | LYM# | Lymphocyte Count | Continuous (10^9/L) |
| | MON# | Monocyte Count | Continuous (10^9/L) |
| | PLT | Platelet Count | Continuous (10^9/L) |
| | MCV | Mean Corpuscular Volume | Continuous (fL) |
| **Organ & Ions** | BUN | Blood Urea Nitrogen | Continuous (mmol/L) |
| | SCr | Serum Creatinine | Continuous (μmol/L) |
| | ALT | Alanine Aminotransferase | Continuous (U/L) |
| | A/G Ratio | Albumin/Globulin Ratio | Continuous |
| | Cl | Chloride | Continuous (mmol/L) |
| | K | Potassium | Continuous (mmol/L) |

---

## 🚀 Deployment

### Local Environment

# 1. Clone the repository
git clone https://github.com/ZMY-521/CVD-predict.git
cd CVD-predict

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py

### Streamlit Cloud (Recommended for Sharing)

1. Upload this repository to your GitHub account.
2. Go to share.streamlit.io.
3. Connect your GitHub repository.
4. Set the Main file path to app.py.
5. Click Deploy.

---

## 📁 File Structure

CVD-Dual-Track-Web/
├── app.py                 # Main Streamlit application UI & Logic
├── Model_Gatekeeper.pkl   # Triage model (Predicts ODKD status)
├── Model_Track_A.pkl      # Specialized model for ODKD+ population
├── Model_Track_B.pkl      # Specialized model for Non-ODKD population
├── Model_Global.pkl       # Baseline unstratified conventional model
├── requirements.txt       # Python dependencies (streamlit, pandas, scikit-learn, joblib)
└── README.md              # Project documentation

---

## ⚠️ Disclaimer

This application is designed for **research, peer-review, and educational purposes only**. The predicted probabilities and triage suggestions do not constitute professional medical advice, diagnosis, or treatment. It should not be used as a substitute for the clinical judgment of a qualified healthcare provider.

---

## 🧬 System Architecture

- **Machine Learning Framework:** Tree-based ensemble pipelines.
- **Routing Logic:** Patients are evaluated by the Gatekeeper (Cutoff: 0.5). If Positive -> Track A; If Negative -> Track B.
- **Primary Endpoint:** Probability of Cardiovascular Disease (CVD) occurrence.
- **Clinical Action Threshold:** 50% probability is used as the strict cutoff for intervention evaluation (High Risk vs. Standard Risk).
