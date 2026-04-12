import streamlit as st

st.set_page_config(
    page_title="Science of Sleep",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #0f0c29; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #0d1f3c 100%); }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-style: italic;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        color: rgba(232,228,248,0.6);
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 0.5px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        color: rgba(232,228,248,0.5);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #4ecdc4;
        line-height: 1;
    }
    .metric-unit { font-size: 0.85rem; color: rgba(232,228,248,0.4); }
    .insight-box {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #7c6fe0;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        color: rgba(232,228,248,0.85);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .cluster-healthy { border-top: 3px solid #4ecdc4; }
    .cluster-moderate { border-top: 3px solid #7c6fe0; }
    .cluster-deprived { border-top: 3px solid #ff8c69; }
    .stSidebar { background: rgba(15,12,41,0.95) !important; }
    .stSidebar .stMarkdown { color: rgba(232,228,248,0.8); }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 0.5px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.04);
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: rgba(232,228,248,0.6);
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,111,224,0.3) !important;
        color: #c4b9ff !important;
    }
    .rec-card {
        background: rgba(78,205,196,0.08);
        border: 0.5px solid rgba(78,205,196,0.25);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from utils.data_loader import load_and_preprocess
from utils.charts import (
    plot_distribution, plot_donut, plot_scatter_quality_stress,
    plot_correlation_bars, plot_feature_importance, plot_radar,
    plot_cluster_gender, plot_occupation_bars, plot_bmi_sleep,
    plot_sleep_by_occupation
)
from utils.clustering import run_kmeans, get_cluster_profiles
from utils.ml_models import run_random_forest, run_classification_report
from utils.insights import generate_insights

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Science of Sleep")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload your dataset (CSV)",
        type=["csv"],
        help="Upload the Kaggle Sleep Health & Lifestyle dataset"
    )
    st.markdown("---")
    st.markdown("**Filters**")
    gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    age_range = st.slider("Age range", 18, 60, (18, 60))
    n_clusters = st.slider("K-Means clusters", 2, 5, 3)
    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(232,228,248,0.4)'>Data: Kaggle Sleep Health & Lifestyle Dataset · 374 subjects</small>",
        unsafe_allow_html=True
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw, df = load_and_preprocess(uploaded)

# Apply filters
df = df[df["Gender"].isin(gender_filter)]
df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar.")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Science of Sleep</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Mining daily sleep behavior — patterns, clusters & insights from real-world data</div>',
    unsafe_allow_html=True
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg Sleep", f"{df['Sleep Duration'].mean():.1f} h")
k2.metric("Avg Quality", f"{df['Quality of Sleep'].mean():.1f} / 10")
k3.metric("Avg Stress", f"{df['Stress Level'].mean():.1f} / 10")
k4.metric("Avg Activity", f"{df['Physical Activity Level'].mean():.0f} min")
k5.metric("Avg Steps", f"{df['Daily Steps'].mean():,.0f}")
disorder_pct = (df['Sleep Disorder'] != 'None').mean() * 100
k6.metric("Has Disorder", f"{disorder_pct:.0f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Overview", " Clustering", " Correlations",
    " By Occupation", " ML Model", " Insights"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_distribution(df, "Sleep Duration", "Sleep Duration Distribution (hours)", "#7c6fe0"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            plot_distribution(df, "Quality of Sleep", "Sleep Quality Distribution (1–10)", "#4ecdc4"),
            use_container_width=True
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            plot_donut(df, "BMI Category", "BMI Category Breakdown"),
            use_container_width=True
        )
    with c4:
        st.plotly_chart(
            plot_donut(df, "Sleep Disorder", "Sleep Disorder Prevalence"),
            use_container_width=True
        )

    st.plotly_chart(
        plot_scatter_quality_stress(df),
        use_container_width=True
    )

    with st.expander(" Raw data preview"):
        st.dataframe(df_raw.head(50), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("K-Means Clustering Analysis")
    st.markdown(
        "K-Means applied on normalized: sleep duration, quality, stress level, physical activity, and daily steps."
    )

    df_clustered, kmeans_model, scaler = run_kmeans(df, n_clusters=n_clusters)
    profiles = get_cluster_profiles(df_clustered, n_clusters)

    # Cluster cards
    cols = st.columns(n_clusters)
    icons = ["🌙", "💤", "⚡", "🌀", "🔥"]
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
        st.plotly_chart(
            plot_radar(profiles, n_clusters),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            plot_cluster_gender(df_clustered, n_clusters),
            use_container_width=True
        )

    st.plotly_chart(
        plot_scatter_quality_stress(df_clustered, color_col="Cluster"),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Correlation Analysis")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(plot_correlation_bars(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_bmi_sleep(df), use_container_width=True)

    # Heatmap
    numeric_cols = [
        "Age", "Sleep Duration", "Quality of Sleep",
        "Physical Activity Level", "Stress Level",
        "Heart Rate", "Daily Steps"
    ]
    import plotly.figure_factory as ff
    corr_matrix = df[numeric_cols].corr().round(2)
    fig_heat = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.values,
        colorscale="Purp",
        showscale=True,
    )
    fig_heat.update_layout(
        title="Correlation Heatmap — All Numeric Features",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(232,228,248,0.8)", size=11),
        height=460,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BY OCCUPATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Sleep Patterns by Occupation")

    st.plotly_chart(plot_occupation_bars(df), use_container_width=True)
    st.plotly_chart(plot_sleep_by_occupation(df), use_container_width=True)

    # Disorder by occupation
    import plotly.express as px
    occ_disorder = df.groupby(["Occupation", "Sleep Disorder"]).size().reset_index(name="Count")
    fig_occ_d = px.bar(
        occ_disorder, x="Occupation", y="Count", color="Sleep Disorder",
        barmode="stack",
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Machine Learning — Sleep Quality Prediction")
    st.markdown(
        "Random Forest classifier predicts whether sleep quality is **High (≥7)** or **Low (<7)**. "
        "Features: stress level, sleep duration, heart rate, daily steps, physical activity, BMI, age."
    )

    with st.spinner("Training Random Forest model..."):
        rf_model, X_test, y_test, y_pred, feature_names, acc, report_df = run_random_forest(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{acc*100:.1f}%")
    col2.metric("Precision (High)", f"{report_df.loc['High Sleep Quality','precision']:.2f}")
    col3.metric("Recall (High)", f"{report_df.loc['High Sleep Quality','recall']:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_feature_importance(rf_model, feature_names),
            use_container_width=True
        )
    with c2:
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        import plotly.graph_objects as go
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Low Quality", "High Quality"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#1a1040"], [1, "#7c6fe0"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted", yaxis_title="Actual",
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

    # Interactive prediction
    st.markdown("---")
    st.markdown("###  Predict your sleep quality")
    p1, p2, p3, p4 = st.columns(4)
    u_stress = p1.slider("Stress level", 1, 10, 5)
    u_duration = p2.slider("Sleep duration (h)", 5.0, 9.0, 7.0, 0.1)
    u_hr = p3.slider("Heart rate (bpm)", 60, 90, 72)
    u_steps = p4.slider("Daily steps", 2000, 12000, 6000, 500)
    p5, p6, p7, _ = st.columns(4)
    u_activity = p5.slider("Physical activity (min)", 10, 120, 45)
    u_bmi = p6.selectbox("BMI category", ["Normal", "Overweight", "Obese"])
    u_age = p7.slider("Age", 18, 65, 35)

    bmi_map = {"Normal": 0, "Overweight": 1, "Obese": 2}
    user_input = np.array([[u_stress, u_duration, u_hr, u_steps, u_activity, bmi_map[u_bmi], u_age]])
    pred = rf_model.predict(user_input)[0]
    prob = rf_model.predict_proba(user_input)[0]

    if pred == "High Sleep Quality":
        st.success(f" Predicted: **High Sleep Quality** — confidence {prob.max()*100:.0f}%")
    else:
        st.error(f" Predicted: **Low Sleep Quality** — confidence {prob.max()*100:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Data-Driven Insights & Recommendations")
    insights = generate_insights(df)
    for ins in insights:
        icon = ins.get("icon")
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
