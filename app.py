import streamlit as st
import pandas as pd
import joblib

# ================= 1. 页面全局配置 =================
st.set_page_config(page_title="Precision CVD Risk Triage")

st.markdown(
    """<style>.stApp { background-color: #FFFFFF; } .main-header {font-size: 2.2rem; font-weight: 800; color: #111827; margin-bottom: 0px; border-bottom: 3px solid #E5E7EB; padding-bottom: 5px;} .sub-header {font-size: 1.2rem; font-weight: 700; color: #4B5563; margin-top: 5px; margin-bottom: 10px;}</style>""",
    unsafe_allow_html=True)


# ================= 2. 顶级游标卡尺 (大幅压缩上下留白) =================
def draw_academic_risk_bar(model_name, subtitle_html, probability, is_dual_track=False):
    prob_pct = probability * 100

    if prob_pct >= 50:
        bar_color, status_text = "#9F1239", "High Risk"
    elif prob_pct >= 30:
        bar_color, status_text = "#B45309", "Intermediate"
    else:
        bar_color, status_text = "#0D9488", "Low Risk"

    title_display = f"✨ {model_name}" if is_dual_track else f"⚙️ {model_name}"

    # 注意这里的 margin-bottom 只有 8px，并且 padding 被极限压缩
    html = f"""<div style="background-color: #FFFFFF; border: 2px solid #E5E7EB; border-radius: 8px; padding: 12px 25px; margin-bottom: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
<h3 style="margin-top: 0; color: #111827; font-weight: 800; font-family: sans-serif; margin-bottom: 4px; font-size: 1.45rem;">{title_display}</h3>
{subtitle_html}
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px; margin-top: 2px;">
<span style="font-size: 2.6rem; font-weight: 800; color: {bar_color}; line-height: 1;">{prob_pct:.1f}%</span>
<span style="font-size: 1.3rem; font-weight: 800; color: {bar_color};">{status_text}</span>
</div>
<div style="position: relative; width: 100%; height: 32px; border-radius: 4px; display: flex; margin-bottom: 12px;">
<div style="width: 30%; height: 100%; background-color: #14B8A6;"></div>
<div style="width: 20%; height: 100%; background-color: #D97706;"></div>
<div style="width: 50%; height: 100%; background-color: #9F1239;"></div>
<div style="position: absolute; left: calc({prob_pct}% - 10px); top: -10px; width: 20px; height: 20px; background-color: #111827; border-radius: 50%; z-index: 20; box-shadow: 0 2px 4px rgba(0,0,0,0.4); border: 2px solid #FFFFFF;"></div>
<div style="position: absolute; left: calc({prob_pct}% - 2px); top: -8px; width: 4px; height: 48px; background-color: #111827; z-index: 19; border-radius: 2px;"></div>
</div>
<div style="position: relative; width: 100%; height: 16px; font-size: 1rem; color: #6B7280; font-weight: 800; font-family: sans-serif;">
<span style="position: absolute; left: 0;">0%</span>
<span style="position: absolute; left: 30%; transform: translateX(-50%);">30%</span>
<span style="position: absolute; left: 50%; transform: translateX(-50%); color: #111827; font-weight: 900;">50%</span>
<span style="position: absolute; right: 0;">100%</span>
</div>
</div>"""
    return html


# ================= 3. 模型加载 =================
@st.cache_resource
def load_models():
    try:
        gatekeeper = joblib.load('Model_Gatekeeper.pkl')
        track_a = joblib.load('Model_Track_A.pkl')
        track_b = joblib.load('Model_Track_B.pkl')
        global_model = joblib.load('Model_Global.pkl')
        return gatekeeper, track_a, track_b, global_model
    except Exception as e:
        return None, None, None, None


gatekeeper, track_a, track_b, global_model = load_models()

st.markdown('<div class="main-header">Precision CVD Risk Stratification: A Heterogeneity-Aware Dual-Track System</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive clinical demonstration highlighting reclassification benefits (NRI/IDI) over unstratified global models.</div>',
    unsafe_allow_html=True)

if gatekeeper is not None:
    st.sidebar.markdown("### 📋 Clinical Input")
    with st.sidebar.expander("👤 Basic & Metabolic", expanded=True):
        age = st.number_input("Age", 18.0, 100.0, 60.0)
        hba1c = st.number_input("HbA1c", 4.0, 15.0, 6.5)
        ua = st.number_input("SUA", 100.0, 800.0, 350.0)
        non_hdl = st.number_input("Non-HDL-C", 0.0, 10.0, 3.5)

    with st.sidebar.expander("🩸 Complete Blood Count", expanded=True):
        rdw = st.number_input("RDW", 10.0, 25.0, 13.5)
        neu = st.number_input("NEU#", 0.0, 20.0, 4.0)
        lym = st.number_input("LYM#", 0.0, 10.0, 1.5)
        mon = st.number_input("MON#", 0.0, 5.0, 0.4)
        plt = st.number_input("PLT", 50.0, 500.0, 200.0)
        mcv = st.number_input("MCV", 60.0, 120.0, 90.0)

    with st.sidebar.expander("🧪 Organ Function & Ions", expanded=True):
        bun = st.number_input("BUN", 0.0, 50.0, 6.5)
        scr = st.number_input("SCr", 30.0, 300.0, 80.0)
        alt = st.number_input("ALT", 0.0, 200.0, 25.0)
        ag_ratio = st.number_input("A/G Ratio", 0.5, 3.0, 1.5)
        cl = st.number_input("Cl", 80.0, 120.0, 100.0)
        k = st.number_input("K", 2.0, 7.0, 4.2)

    df_gate = pd.DataFrame([[age, bun, rdw, ua, hba1c, cl, ag_ratio, neu]],
                           columns=['Age', 'BUN', 'RDW', 'SUA', 'HbA1c', 'Cl', 'A/G', 'NEU#'])
    df_track_a = pd.DataFrame([[alt, age, rdw, non_hdl, plt, hba1c, cl, scr]],
                              columns=['ALT', 'Age', 'RDW', 'Non-HDL-C', 'PLT', 'HbA1c', 'Cl', 'SCr'])
    df_track_b = pd.DataFrame([[ua, age, k, rdw, non_hdl, mcv, scr]],
                              columns=['SUA', 'Age', 'K', 'RDW', 'Non-HDL-C', 'MCV', 'SCr'])
    df_global = pd.DataFrame([[ua, alt, mon, age, lym, rdw, non_hdl, k, plt, mcv, scr]],
                             columns=['SUA', 'ALT', 'MON#', 'Age', 'LYM#', 'RDW', 'Non-HDL-C', 'K', 'PLT', 'MCV',
                                      'SCr'])

    if st.sidebar.button("🔍 Run Prediction", use_container_width=True, type="primary"):
        global_risk = global_model.predict_proba(df_global)[0][1]
        is_odkd_positive = gatekeeper.predict_proba(df_gate)[0][1] > 0.5

        # 将四个模块全部合并成一个巨型 HTML 字符串，彻底消灭 Streamlit 注入的间隙！
        final_html_block = ""

        # 1. 全局模型
        global_subtitle = """<p style="color: #4B5563; font-size: 1.1rem; font-weight: 600; margin-bottom: 2px;">Unstratified baseline approach without heterogeneity assessment.</p>"""
        final_html_block += draw_academic_risk_bar("Global Model", global_subtitle, global_risk, False)

        # 2. 盾牌
        if is_odkd_positive:
            dual_track_risk = track_a.predict_proba(df_track_a)[0][1]
            shield_html = """<div style="display: flex; align-items: center; background-color: #F8FAFC; padding: 8px 15px; border-radius: 6px; border-left: 6px solid #9F1239; margin-bottom: 8px;">
<span style="font-size: 1.8rem; margin-right: 15px;">🛡️</span>
<div><h5 style="margin: 0; color: #1E293B; font-weight: 800; font-size: 1.15rem;">Gatekeeper Triage Active</h5>
<p style="margin: 0; color: #475569; font-size: 1.05rem; font-weight: 600;">Patient classified as <b>ODKD</b> ➔ Auto-routing to <b>Track A</b></p></div></div>"""
        else:
            dual_track_risk = track_b.predict_proba(df_track_b)[0][1]
            shield_html = """<div style="display: flex; align-items: center; background-color: #F8FAFC; padding: 8px 15px; border-radius: 6px; border-left: 6px solid #14B8A6; margin-bottom: 8px;">
<span style="font-size: 1.8rem; margin-right: 15px;">🛡️</span>
<div><h5 style="margin: 0; color: #1E293B; font-weight: 800; font-size: 1.15rem;">Gatekeeper Triage Active</h5>
<p style="margin: 0; color: #475569; font-size: 1.05rem; font-weight: 600;">Patient classified as <b>Non-ODKD</b> ➔ Auto-routing to <b>Track B</b></p></div></div>"""
        final_html_block += shield_html

        # 3. 双轨模型
        dt_subtitle = ""  # 移除重复的副标题，让排版更紧实
        final_html_block += draw_academic_risk_bar("Dual-Track System", dt_subtitle, dual_track_risk, True)

        # 4. 结论区
        if global_risk >= 0.5 and dual_track_risk < 0.5:
            conclusion_html = """<div style="border-left: 6px solid #14B8A6; background-color: #F0FDFA; padding: 12px 20px; border-radius: 0 6px 6px 0; margin-bottom: 0px;">
<h3 style="color: #0F766E; margin: 0 0 4px 0; font-size: 1.35rem;">🟢 Precision Downgrading (NRI Benefit)</h3>
<p style="margin: 0; color: #115E59; font-size: 1.1rem; font-weight: 600; line-height: 1.3;"><b>Mechanism:</b> Overestimation corrected. Patient safely downgraded below 50% cutoff, preventing overtreatment.</p>
</div>"""

        elif global_risk < 0.5 and dual_track_risk >= 0.5:
            conclusion_html = """<div style="border-left: 6px solid #9F1239; background-color: #FFF1F2; padding: 12px 20px; border-radius: 0 6px 6px 0; margin-bottom: 0px;">
<h3 style="color: #881337; margin: 0 0 4px 0; font-size: 1.35rem;">🔴 Precision Upgrading (IDI Benefit)</h3>
<p style="margin: 0; color: #4C0519; font-size: 1.1rem; font-weight: 600; line-height: 1.3;"><b>Mechanism:</b> False negative rescued. Patient correctly upgraded above 50% cutoff, enabling timely intervention.</p>
</div>"""

        else:
            conclusion_html = """<div style="border-left: 6px solid #6B7280; background-color: #F9FAFB; padding: 12px 20px; border-radius: 0 6px 6px 0; margin-bottom: 0px;">
<h3 style="color: #374151; margin: 0 0 4px 0; font-size: 1.35rem;">⚖️ Risk Consensus (No Reclassification)</h3>
<p style="margin: 0; color: #4B5563; font-size: 1.1rem; font-weight: 600; line-height: 1.3;"><b>Mechanism:</b> Both models agree on the primary clinical action threshold. No cross-threshold reclassification triggered.</p>
</div>"""
        final_html_block += conclusion_html

        # 唯一的一次渲染调用，彻底切断 Streamlit 强行加空隙的可能！
        st.markdown(f"<div>{final_html_block}</div>", unsafe_allow_html=True)