import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Dual-Track CVD Risk Predictor", page_icon="🫀", layout="wide")
st.title("🫀 AI-Powered Dual-Track Triage System vs. Global Model")
st.markdown("**Real-time Demonstration of Precision Risk Reclassification**")
st.markdown("---")


@st.cache_resource
def load_models():
    try:
        gatekeeper = joblib.load('Model_Gatekeeper.pkl')
        track_a = joblib.load('Model_Track_A.pkl')
        track_b = joblib.load('Model_Track_B.pkl')
        global_model = joblib.load('Model_Global.pkl')
        return gatekeeper, track_a, track_b, global_model
    except Exception as e:
        st.error(f"⚠️ 找不到模型文件，请确保4个.pkl文件都在同一目录下。报错信息: {e}")
        return None, None, None, None


gatekeeper, track_a, track_b, global_model = load_models()

if gatekeeper is not None:
    st.sidebar.header("📋 Input Patient Features")
    st.sidebar.markdown("Please input the unique clinical features:")

    # 获取特征 (提取所有去重后的16个指标，方便医生一次性输入)
    age = st.sidebar.number_input("Age (years)", 18.0, 100.0, 60.0)
    bun = st.sidebar.number_input("BUN (mmol/L)", 0.0, 50.0, 6.5)
    rdw = st.sidebar.number_input("RDW (%)", 10.0, 25.0, 13.5)
    sua = st.sidebar.number_input("SUA (μmol/L)", 100.0, 800.0, 350.0)
    hba1c = st.sidebar.number_input("HbA1c (%)", 4.0, 15.0, 6.5)
    cl = st.sidebar.number_input("Cl (mmol/L)", 80.0, 120.0, 100.0)
    ag_ratio = st.sidebar.number_input("A/G Ratio", 0.5, 3.0, 1.5)
    neu = st.sidebar.number_input("NEU# (10^9/L)", 0.0, 20.0, 4.0)

    alt = st.sidebar.number_input("ALT (U/L)", 0.0, 200.0, 25.0)
    non_hdl = st.sidebar.number_input("Non-HDL-C (mmol/L)", 0.0, 10.0, 3.5)
    plt = st.sidebar.number_input("PLT (10^9/L)", 50.0, 500.0, 200.0)
    scr = st.sidebar.number_input("SCr (μmol/L)", 30.0, 300.0, 80.0)
    k = st.sidebar.number_input("K (Potassium) (mmol/L)", 2.0, 7.0, 4.2)
    mcv = st.sidebar.number_input("MCV (fL)", 60.0, 120.0, 90.0)
    mon = st.sidebar.number_input("MON# (10^9/L)", 0.0, 5.0, 0.4)
    lym = st.sidebar.number_input("LYM# (10^9/L)", 0.0, 10.0, 1.5)

    # ================= 严格按照你的要求构建 DataFrame =================
    # Gatekeeper: Age, BUN, RDW, SUA, HbA1c, Cl, A/G, NEU#
    df_gate = pd.DataFrame([[age, bun, rdw, sua, hba1c, cl, ag_ratio, neu]],
                           columns=['Age', 'BUN', 'RDW', 'SUA', 'HbA1c', 'Cl', 'A/G', 'NEU#'])

    # Track A: ALT, Age, RDW, Non-HDL-C, PLT, HbA1c, Cl, SCr
    df_track_a = pd.DataFrame([[alt, age, rdw, non_hdl, plt, hba1c, cl, scr]],
                              columns=['ALT', 'Age', 'RDW', 'Non-HDL-C', 'PLT', 'HbA1c', 'Cl', 'SCr'])

    # Track B: SUA, Age, K, RDW, Non-HDL-C, MCV, SCr
    df_track_b = pd.DataFrame([[sua, age, k, rdw, non_hdl, mcv, scr]],
                              columns=['SUA', 'Age', 'K', 'RDW', 'Non-HDL-C', 'MCV', 'SCr'])

    # Global: SUA, ALT, MON#, Age, LYM#, RDW, Non-HDL-C, K, PLT, MCV, SCr
    df_global = pd.DataFrame([[sua, alt, mon, age, lym, rdw, non_hdl, k, plt, mcv, scr]],
                             columns=['SUA', 'ALT', 'MON#', 'Age', 'LYM#', 'RDW', 'Non-HDL-C', 'K', 'PLT', 'MCV',
                                      'SCr'])

    # ================= 对决运算引擎 =================
    st.markdown("### Click below to run both systems simultaneously:")
    if st.button("🚀 Run Dual-Track Triage vs Global Model", use_container_width=True):

        global_risk = global_model.predict_proba(df_global)[0][1]

        odkd_prob = gatekeeper.predict_proba(df_gate)[0][1]
        is_odkd_positive = odkd_prob > 0.5

        if is_odkd_positive:
            dual_track_risk = track_a.predict_proba(df_track_a)[0][1]
            track_name = "Track A (ODKD Positive)"
            track_color = "🔴"
        else:
            dual_track_risk = track_b.predict_proba(df_track_b)[0][1]
            track_name = "Track B (ODKD Negative)"
            track_color = "🔵"

        # ================= 终极对决 UI 展示 =================
        st.markdown("---")
        st.markdown("### 🏆 Head-to-Head Comparison")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⚙️ Conventional Global Model")
            st.metric(label="Predicted CVD Risk", value=f"{global_risk:.1%}")
            if global_risk > 0.5:
                st.error("⚠️ Status: High Risk")
            else:
                st.success("✅ Status: Standard Risk")

        with col2:
            st.markdown("#### ✨ Proposed Dual-Track System")
            st.info(f"{track_color} Triage Result: Auto-Routed to **{track_name}**")
            st.metric(label="Predicted CVD Risk", value=f"{dual_track_risk:.1%}")
            if dual_track_risk > 0.5:
                st.error("⚠️ Status: High Risk")
            else:
                st.success("✅ Status: Standard Risk")

        # ================= NRI / IDI 获益动态捕获 =================
        st.markdown("---")
        st.markdown("### 💡 Clinical Reclassification Impact")

        if global_risk > 0.5 and dual_track_risk <= 0.5:
            st.success(
                "🎉 **Precision Downgrading (NRI Benefit):** The Global Model overestimated the risk. The Dual-Track system correctly downgraded this patient to Standard Risk, potentially **avoiding overtreatment**.")
        elif global_risk <= 0.5 and dual_track_risk > 0.5:
            st.warning(
                "🚨 **Precision Upgrading (IDI Benefit):** The Global Model missed this high-risk patient. The Dual-Track system correctly upgraded this patient to High Risk, enabling **timely clinical intervention**.")
        else:
            st.write("⚖️ Both models agree on the broad risk category for this specific parameter combination.")