import streamlit as st

st.set_page_config(
    page_title="Science of Sleep",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #0f0c29; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #0d1f3c 100%); }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem; font-style: italic;
        color: #ffffff; line-height: 1.1; margin-bottom: 0.3rem;
    }
    .hero-sub { color: rgba(232,228,248,0.6); font-size: 1rem; margin-bottom: 1rem; }
    .stSidebar { background: rgba(15,12,41,0.95) !important; }
    .stSidebar .stMarkdown { color: rgba(232,228,248,0.8); }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 0.5px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: rgba(255,255,255,0.04);
        padding: 4px; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px; color: rgba(232,228,248,0.6); font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,111,224,0.3) !important; color: #c4b9ff !important;
    }
    .tip-card {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #4ecdc4;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.1rem; margin: 0.5rem 0;
        color: rgba(232,228,248,0.85); font-size: 0.88rem; line-height: 1.6;
    }
    .tip-card.warn { border-left-color: #ff8c69; }
    .tip-card.info { border-left-color: #7c6fe0; }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import load_and_preprocess
from utils.charts import (
    plot_distribution, plot_donut, plot_scatter_quality_stress,
    plot_correlation_bars, plot_feature_importance, plot_radar,
    plot_cluster_gender, plot_occupation_bars, plot_bmi_sleep,
    plot_sleep_by_occupation
)
from utils.clustering import run_kmeans, get_cluster_profiles, CLUSTER_FEATURES
from utils.ml_models import run_random_forest
from utils.insights import generate_insights

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌙 Science of Sleep")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["📊 Dataset Analysis", "👤 My Sleep Data"],
        help="Switch between analysing a full dataset or entering your own personal sleep data."
    )
    st.markdown("---")

    if mode == "📊 Dataset Analysis":
        uploaded = st.file_uploader(
            "Upload dataset (CSV)",
            type=["csv"],
            help="Upload the Kaggle Sleep Health & Lifestyle CSV"
        )
        st.markdown("**Filters**")
        gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
        age_range     = st.slider("Age range", 18, 60, (18, 60))
        n_clusters    = st.slider("K-Means clusters", 2, 5, 3)
        st.markdown("---")
        st.markdown(
            "<small style='color:rgba(232,228,248,0.4)'>Kaggle Sleep Health & Lifestyle Dataset</small>",
            unsafe_allow_html=True
        )
        individual = None

    else:
        st.markdown("**Personal info**")
        ind_name       = st.text_input("Your name", value="User")
        ind_age        = st.slider("Age", 18, 65, 25)
        ind_gender     = st.selectbox("Gender", ["Male", "Female"])
        ind_occupation = st.selectbox("Occupation", [
            "Software Engineer", "Doctor", "Nurse", "Teacher",
            "Accountant", "Lawyer", "Engineer", "Sales Representative",
            "Manager", "Other"
        ])
        st.markdown("**Sleep habits**")
        ind_duration = st.slider("Sleep duration (hours/night)", 3.0, 12.0, 7.0, 0.5)
        ind_quality  = st.slider("Sleep quality you feel (1–10)", 1, 10, 7)
        ind_stress   = st.slider("Stress level (1–10)", 1, 10, 5)
        st.markdown("**Activity & health**")
        ind_activity = st.slider("Physical activity (min/day)", 0, 180, 45)
        ind_steps    = st.slider("Daily steps", 1000, 15000, 6000, 500)
        ind_bmi      = st.selectbox("BMI category", ["Normal", "Overweight", "Obese"])
        ind_hr       = st.slider("Resting heart rate (bpm)", 50, 100, 72)
        ind_disorder = st.selectbox("Known sleep disorder", ["None", "Insomnia", "Sleep Apnea"])

        individual = {
            "name": ind_name, "age": ind_age, "gender": ind_gender,
            "occupation": ind_occupation, "duration": ind_duration,
            "quality": ind_quality, "stress": ind_stress,
            "activity": ind_activity, "steps": ind_steps,
            "bmi": ind_bmi, "hr": ind_hr, "disorder": ind_disorder
        }
        uploaded      = None
        gender_filter = ["Male", "Female"]
        age_range     = (18, 65)
        n_clusters    = 3

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Science of Sleep</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Mining daily sleep behavior — patterns, clusters & insights</div>',
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# ── INDIVIDUAL MODE
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "👤 My Sleep Data":
    d = individual

    # Train model on background dataset
    _, df_bg = load_and_preprocess(None)
    rf_model, _, _, _, feature_names, _, _ = run_random_forest(df_bg)

    bmi_map   = {"Normal": 0, "Overweight": 1, "Obese": 2}
    user_arr  = np.array([[
        d["stress"], d["duration"], d["hr"],
        d["steps"],  d["activity"], bmi_map[d["bmi"]], d["age"]
    ]])
    pred  = rf_model.predict(user_arr)[0]
    proba = rf_model.predict_proba(user_arr)[0]
    conf  = proba.max() * 100

    # Composite sleep score 0-100
    sleep_score = int(
        (d["quality"]               / 10) * 40 +
        (min(d["duration"], 9)      /  9) * 25 +
        ((10 - d["stress"])         /  9) * 20 +
        (min(d["activity"], 90)     / 90) * 15
    )

    if sleep_score >= 75:
        score_color, score_label, score_emoji = "#4ecdc4", "Excellent", "🌙"
    elif sleep_score >= 55:
        score_color, score_label, score_emoji = "#7c6fe0", "Moderate", "💤"
    else:
        score_color, score_label, score_emoji = "#ff8c69", "Needs Work", "⚡"

    # ── Profile card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05);border:0.5px solid rgba(255,255,255,0.12);
        border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem">
        <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
            <div style="width:64px;height:64px;border-radius:50%;
                background:rgba(124,111,224,0.2);border:2px solid #7c6fe0;
                display:flex;align-items:center;justify-content:center;font-size:1.8rem">
                {score_emoji}</div>
            <div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;
                    color:#fff;margin-bottom:0.2rem">{d['name']}</div>
                <div style="color:rgba(232,228,248,0.5);font-size:0.85rem">
                    {d['gender']} · Age {d['age']} · {d['occupation']} · BMI: {d['bmi']}
                </div>
                <div style="margin-top:8px">
                    <span style="display:inline-block;padding:5px 14px;border-radius:20px;
                        border:1px solid {score_color};color:{score_color};font-size:0.88rem;
                        background:rgba(255,255,255,0.04);margin-right:8px">
                        Sleep Score: {sleep_score}/100 — {score_label}
                    </span>
                    <span style="display:inline-block;padding:5px 14px;border-radius:20px;
                        border:1px solid rgba(255,255,255,0.15);color:rgba(232,228,248,0.7);
                        font-size:0.88rem;background:rgba(255,255,255,0.04)">
                        Disorder: {d['disorder']}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Sleep Duration",    f"{d['duration']} h")
    k2.metric("Sleep Quality",     f"{d['quality']} / 10")
    k3.metric("Stress Level",      f"{d['stress']} / 10")
    k4.metric("Physical Activity", f"{d['activity']} min")
    k5.metric("Daily Steps",       f"{d['steps']:,}")
    k6.metric("Heart Rate",        f"{d['hr']} bpm")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ML prediction banner ──────────────────────────────────────────────────
    if pred == "High Sleep Quality":
        st.success(f"✅ ML Prediction: **High Sleep Quality** — model confidence {conf:.0f}%")
    else:
        st.error(f"⚠️ ML Prediction: **Low Sleep Quality** — model confidence {conf:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar + Tips side by side ─────────────────────────────────────────────
    col_chart, col_tips = st.columns([1, 1])

    with col_chart:
        cats      = ["Sleep Duration", "Sleep Quality", "Low Stress", "Activity", "Steps"]
        vals_user = [
            (d["duration"]           /  9) * 100,
            (d["quality"]            / 10) * 100,
            ((10 - d["stress"])      /  9) * 100,
            (min(d["activity"], 90)  / 90) * 100,
            (min(d["steps"], 10000)  / 10000) * 100,
        ]
        vals_avg = [79, 73, 46, 65, 68]   # dataset averages (normalised)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals_user + [vals_user[0]], theta=cats + [cats[0]],
            name="You", fill="toself",
            fillcolor="rgba(124,111,224,0.15)",
            line=dict(color="#7c6fe0", width=2),
            marker=dict(color="#7c6fe0", size=7)
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=vals_avg + [vals_avg[0]], theta=cats + [cats[0]],
            name="Dataset avg", fill="toself",
            fillcolor="rgba(78,205,196,0.08)",
            line=dict(color="#4ecdc4", width=2, dash="dot"),
            marker=dict(color="#4ecdc4", size=5)
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    color="rgba(232,228,248,0.3)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(size=9, color="rgba(232,228,248,0.4)")
                ),
                angularaxis=dict(
                    color="rgba(232,228,248,0.5)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(size=10)
                )
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(232,228,248,0.8)", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title="Your Profile vs Dataset Average",
            height=380, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_tips:
        st.markdown("#### 💡 Personalised recommendations")
        tips = []

        if d["duration"] < 7:
            tips.append(("warn",
                f"You're sleeping only {d['duration']}h. Adults need 7–9 hours. "
                "Try going to bed 30 min earlier each night."))
        else:
            tips.append(("ok",
                f"Great — {d['duration']}h is within the recommended 7–9h range."))

        if d["stress"] >= 7:
            tips.append(("warn",
                f"Stress level {d['stress']}/10 is high. Stress is the strongest predictor "
                "of poor sleep (r = −0.87). Try mindfulness or a wind-down routine before bed."))
        elif d["stress"] >= 5:
            tips.append(("info",
                f"Moderate stress ({d['stress']}/10). Small reductions — a 10-min walk or "
                "deep breathing — can noticeably improve sleep quality."))
        else:
            tips.append(("ok", f"Stress level {d['stress']}/10 is well managed. Keep it up!"))

        if d["activity"] < 30:
            tips.append(("warn",
                f"Only {d['activity']} min/day of activity. People exercising ≥60 min/day "
                "score 1.5 points higher on sleep quality on average."))
        elif d["activity"] < 60:
            tips.append(("info",
                f"{d['activity']} min/day is good — pushing toward 60 min could improve sleep further."))
        else:
            tips.append(("ok",
                f"Excellent — {d['activity']} min/day puts you in the top sleep health group."))

        if d["steps"] < 5000:
            tips.append(("warn",
                f"Only {d['steps']:,} steps/day. Aim for 7,000+ — "
                "steps have a +0.43 correlation with sleep quality."))

        if d["bmi"] in ["Overweight", "Obese"]:
            tips.append(("info",
                f"BMI '{d['bmi']}' is associated with higher sleep apnea risk. "
                "Weight management can significantly improve sleep quality."))

        if d["hr"] >= 80:
            tips.append(("warn",
                f"Resting HR {d['hr']} bpm is elevated (r = −0.67 with sleep quality). "
                "Regular aerobic exercise lowers resting HR over time."))

        for style, text in tips:
            css = "tip-card warn" if style == "warn" else ("tip-card info" if style == "info" else "tip-card")
            st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── You vs dataset comparison bar ─────────────────────────────────────────
    st.markdown("#### 📊 How you compare to the dataset")
    _, df_bg2 = load_and_preprocess(None)
    compare = {
        "Sleep Duration (h)":  (d["duration"],  df_bg2["Sleep Duration"].mean()),
        "Sleep Quality (/10)": (d["quality"],   df_bg2["Quality of Sleep"].mean()),
        "Stress Level (/10)":  (d["stress"],    df_bg2["Stress Level"].mean()),
        "Activity (min)":      (d["activity"],  df_bg2["Physical Activity Level"].mean()),
        "Heart Rate (bpm)":    (d["hr"],        df_bg2["Heart Rate"].mean()),
    }
    lbl   = list(compare.keys())
    u_v   = [v[0] for v in compare.values()]
    avg_v = [round(v[1], 1) for v in compare.values()]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="You", x=lbl, y=u_v,
        marker=dict(color="#7c6fe0", line=dict(width=0)),
        text=[str(v) for v in u_v], textposition="outside",
        textfont=dict(color="rgba(232,228,248,0.8)", size=11)
    ))
    fig_cmp.add_trace(go.Bar(
        name="Dataset avg", x=lbl, y=avg_v,
        marker=dict(color="#4ecdc4", line=dict(width=0)),
        text=[str(v) for v in avg_v], textposition="outside",
        textfont=dict(color="rgba(232,228,248,0.8)", size=11)
    ))
    fig_cmp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(232,228,248,0.8)", family="DM Sans", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=360, margin=dict(l=10, r=10, t=30, b=20)
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Cluster placement ─────────────────────────────────────────────────────
    st.markdown("#### 🔬 Your cluster placement")
    df_clustered, km_model, km_scaler = run_kmeans(df_bg2, n_clusters=3)

    user_cluster_input = np.array([[
        d["duration"], d["quality"], d["stress"], d["activity"], d["steps"]
    ]])
    user_scaled     = km_scaler.transform(user_cluster_input)
    user_cluster_raw= km_model.predict(user_scaled)[0]

    quality_means   = df_clustered.groupby("Cluster")["Quality of Sleep"].mean()
    rank_map        = {old: new for new, old in
                       enumerate(quality_means.sort_values(ascending=False).index)}
    user_cluster    = rank_map[user_cluster_raw]

    c_names  = ["Healthy Sleeper 🌙", "Moderate Sleeper 💤", "Sleep-Deprived ⚡"]
    c_colors = ["#4ecdc4", "#7c6fe0", "#ff8c69"]
    cname    = c_names[user_cluster]  if user_cluster < len(c_names)  else f"Cluster {user_cluster+1}"
    ccolor   = c_colors[user_cluster] if user_cluster < len(c_colors) else "#a8dadc"

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border:1px solid {ccolor};
        border-radius:12px;padding:1.2rem 1.5rem;text-align:center;margin-bottom:1rem">
        <div style="font-size:0.75rem;color:rgba(232,228,248,0.5);
            text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px">
            Based on K-Means clustering, you belong to</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:{ccolor}">
            {cname}</div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()  # Skip dataset tabs in individual mode


# ═══════════════════════════════════════════════════════════════════════════════
# ── DATASET MODE
# ═══════════════════════════════════════════════════════════════════════════════
df_raw, df = load_and_preprocess(uploaded)
df = df[df["Gender"].isin(gender_filter)]
df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar.")
    st.stop()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg Sleep",    f"{df['Sleep Duration'].mean():.1f} h")
k2.metric("Avg Quality",  f"{df['Quality of Sleep'].mean():.1f} / 10")
k3.metric("Avg Stress",   f"{df['Stress Level'].mean():.1f} / 10")
k4.metric("Avg Activity", f"{df['Physical Activity Level'].mean():.0f} min")
k5.metric("Avg Steps",    f"{df['Daily Steps'].mean():,.0f}")
k6.metric("Has Disorder", f"{(df['Sleep Disorder'] != 'None').mean()*100:.0f}%")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔬 Clustering", "🔗 Correlations",
    "💼 By Occupation", "🤖 ML Model", "💡 Insights"
])

# TAB 1 — OVERVIEW
with tab1:
    st.subheader("Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_distribution(df, "Sleep Duration", "Sleep Duration Distribution (hours)", "#7c6fe0"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_distribution(df, "Quality of Sleep", "Sleep Quality Distribution (1–10)", "#4ecdc4"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_donut(df, "BMI Category", "BMI Category Breakdown"), use_container_width=True)
    with c4:
        st.plotly_chart(plot_donut(df, "Sleep Disorder", "Sleep Disorder Prevalence"), use_container_width=True)
    st.plotly_chart(plot_scatter_quality_stress(df), use_container_width=True)
    with st.expander("📋 Raw data preview"):
        st.dataframe(df_raw.head(50), use_container_width=True)

# TAB 2 — CLUSTERING
with tab2:
    st.subheader("K-Means Clustering Analysis")
    st.markdown("K-Means applied on normalized: sleep duration, quality, stress level, physical activity, and daily steps.")
    df_clustered, kmeans_model, scaler = run_kmeans(df, n_clusters=n_clusters)
    profiles = get_cluster_profiles(df_clustered, n_clusters)
    cols = st.columns(n_clusters)
    icons      = ["🌙", "💤", "⚡", "🌀", "🔥"]
    colors_cls = ["#4ecdc4", "#7c6fe0", "#ff8c69", "#ffd166", "#a8dadc"]
    for i, col in enumerate(cols):
        p = profiles[i]
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border:0.5px solid rgba(255,255,255,0.1);
                border-top:3px solid {colors_cls[i]};border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:2rem">{icons[i]}</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#fff;margin:8px 0">
                    Cluster {i+1}</div>
                <div style="font-size:1.8rem;color:{colors_cls[i]};font-family:'DM Serif Display',serif">
                    {p['pct']:.0f}%</div>
                <hr style="border-color:rgba(255,255,255,0.08);margin:10px 0">
                <div style="font-size:0.8rem;color:rgba(232,228,248,0.6);text-align:left">
                    Sleep: <b style="color:#fff">{p['sleep']:.1f}h</b><br>
                    Quality: <b style="color:#fff">{p['quality']:.1f}/10</b><br>
                    Stress: <b style="color:#fff">{p['stress']:.1f}/10</b><br>
                    Activity: <b style="color:#fff">{p['activity']:.0f} min</b><br>
                    Steps: <b style="color:#fff">{p['steps']:,.0f}/day</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_radar(profiles, n_clusters), use_container_width=True)
    with c2:
        st.plotly_chart(plot_cluster_gender(df_clustered, n_clusters), use_container_width=True)
    st.plotly_chart(plot_scatter_quality_stress(df_clustered, color_col="Cluster"), use_container_width=True)

# TAB 3 — CORRELATIONS
with tab3:
    st.subheader("Correlation Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_correlation_bars(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_bmi_sleep(df), use_container_width=True)
    import plotly.figure_factory as ff
    numeric_cols = ["Age", "Sleep Duration", "Quality of Sleep",
                    "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps"]
    corr_matrix = df[numeric_cols].corr().round(2)
    fig_heat = ff.create_annotated_heatmap(
        z=corr_matrix.values, x=list(corr_matrix.columns),
        y=list(corr_matrix.index), annotation_text=corr_matrix.values,
        colorscale="Purp", showscale=True,
    )
    fig_heat.update_layout(
        title="Correlation Heatmap — All Numeric Features",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(232,228,248,0.8)", size=11),
        height=460, margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# TAB 4 — BY OCCUPATION
with tab4:
    st.subheader("Sleep Patterns by Occupation")
    st.plotly_chart(plot_occupation_bars(df), use_container_width=True)
    st.plotly_chart(plot_sleep_by_occupation(df), use_container_width=True)
    occ_disorder = df.groupby(["Occupation", "Sleep Disorder"]).size().reset_index(name="Count")
    fig_occ_d = px.bar(
        occ_disorder, x="Occupation", y="Count", color="Sleep Disorder", barmode="stack",
        color_discrete_map={"None": "#4ecdc4", "Insomnia": "#7c6fe0", "Sleep Apnea": "#ff8c69"},
        title="Sleep Disorder Distribution by Occupation"
    )
    fig_occ_d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(232,228,248,0.8)"),
        xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=380, margin=dict(l=20, r=20, t=50, b=60)
    )
    st.plotly_chart(fig_occ_d, use_container_width=True)

# TAB 5 — ML MODEL
with tab5:
    st.subheader("Machine Learning — Sleep Quality Prediction")
    st.markdown(
        "Random Forest classifier predicts whether sleep quality is **High (≥7)** or **Low (<7)**. "
        "Features: stress level, sleep duration, heart rate, daily steps, physical activity, BMI, age."
    )
    with st.spinner("Training Random Forest model..."):
        rf_model, X_test, y_test, y_pred, feature_names, acc, report_df = run_random_forest(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy",   f"{acc*100:.1f}%")
    col2.metric("Precision (High)", f"{report_df.loc['High Sleep Quality','precision']:.2f}")
    col3.metric("Recall (High)",    f"{report_df.loc['High Sleep Quality','recall']:.2f}")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_feature_importance(rf_model, feature_names), use_container_width=True)
    with c2:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Low Quality", "High Quality"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#1a1040"], [1, "#7c6fe0"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color="white"), showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(232,228,248,0.8)"),
            height=360, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("**Classification Report**")
    st.dataframe(
        report_df.style.background_gradient(cmap="Purples", subset=["precision", "recall", "f1-score"]),
        use_container_width=True
    )
    st.markdown("---")
    st.markdown("### 🔮 Predict your sleep quality")
    p1, p2, p3, p4 = st.columns(4)
    u_stress   = p1.slider("Stress level",            1, 10, 5)
    u_duration = p2.slider("Sleep duration (h)",      5.0, 9.0, 7.0, 0.1)
    u_hr       = p3.slider("Heart rate (bpm)",        60, 90, 72)
    u_steps    = p4.slider("Daily steps",             2000, 12000, 6000, 500)
    p5, p6, p7, _ = st.columns(4)
    u_activity = p5.slider("Physical activity (min)", 10, 120, 45)
    u_bmi      = p6.selectbox("BMI category",         ["Normal", "Overweight", "Obese"])
    u_age      = p7.slider("Age",                     18, 65, 35)
    bmi_map    = {"Normal": 0, "Overweight": 1, "Obese": 2}
    user_input = np.array([[u_stress, u_duration, u_hr, u_steps, u_activity, bmi_map[u_bmi], u_age]])
    pred       = rf_model.predict(user_input)[0]
    prob       = rf_model.predict_proba(user_input)[0]
    if pred == "High Sleep Quality":
        st.success(f"✅ Predicted: **High Sleep Quality** — confidence {prob.max()*100:.0f}%")
    else:
        st.error(f"⚠️ Predicted: **Low Sleep Quality** — confidence {prob.max()*100:.0f}%")

# TAB 6 — INSIGHTS
with tab6:
    st.subheader("Data-Driven Insights & Recommendations")
    insights = generate_insights(df)
    for ins in insights:
        icon  = ins.get("icon",  "💡")
        color = ins.get("color", "#7c6fe0")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-left:3px solid {color};
            border-radius:0 10px 10px 0;padding:1rem 1.2rem;margin:0.6rem 0">
            <div style="font-size:0.95rem;font-weight:500;color:#fff;margin-bottom:6px">
                {icon} {ins['title']}</div>
            <div style="font-size:0.85rem;color:rgba(232,228,248,0.7);line-height:1.7">
                {ins['body']}</div>
            <span style="display:inline-block;font-size:0.7rem;padding:3px 10px;border-radius:12px;
                margin-top:8px;background:rgba(255,255,255,0.06);color:rgba(232,228,248,0.6);
                border:0.5px solid rgba(255,255,255,0.12)">{ins['tag']}</span>
        </div>
        """, unsafe_allow_html=True)
