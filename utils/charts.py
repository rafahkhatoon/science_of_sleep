"""
utils/charts.py
All Plotly visualisation functions for the Sleep dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Theme constants ───────────────────────────────────────────────────────────
PURPLE  = "#7c6fe0"
TEAL    = "#4ecdc4"
ORANGE  = "#ff8c69"
YELLOW  = "#ffd166"
LIGHT   = "#a8dadc"
PINK    = "#f4a261"
COLORS  = [TEAL, PURPLE, ORANGE, YELLOW, LIGHT, PINK]

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(232,228,248,0.8)", family="DM Sans", size=11),
    margin=dict(l=20, r=20, t=50, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
GRID = dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)")


def _apply_layout(fig, title="", height=360):
    fig.update_layout(**LAYOUT, title=title, height=height)
    return fig


# ── 1. Distribution histogram ─────────────────────────────────────────────────
def plot_distribution(df, col, title, color=PURPLE):
    fig = px.histogram(
        df, x=col, nbins=20,
        color_discrete_sequence=[color]
    )
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, title)


# ── 2. Donut chart ────────────────────────────────────────────────────────────
def plot_donut(df, col, title):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    fig = px.pie(
        counts, names=col, values="Count",
        color_discrete_sequence=COLORS,
        hole=0.6
    )
    fig.update_traces(
        textinfo="label+percent",
        textfont=dict(size=11, color="rgba(232,228,248,0.85)"),
        marker=dict(line=dict(color="#0f0c29", width=2))
    )
    return _apply_layout(fig, title)


# ── 3. Scatter: sleep quality vs stress ──────────────────────────────────────
def plot_scatter_quality_stress(df, color_col="Sleep Disorder"):
    if color_col == "Cluster":
        color_map = {
            str(i): COLORS[i] for i in range(10)
        }
        df = df.copy()
        df["Cluster"] = df["Cluster"].astype(str)
        fig = px.scatter(
            df, x="Stress Level", y="Quality of Sleep",
            color="Cluster",
            color_discrete_sequence=COLORS,
            opacity=0.75,
            title="Sleep Quality vs Stress Level — by Cluster",
            labels={"Stress Level": "Stress Level (1–10)", "Quality of Sleep": "Sleep Quality (1–10)"}
        )
    else:
        fig = px.scatter(
            df, x="Stress Level", y="Quality of Sleep",
            color="Sleep Disorder",
            color_discrete_map={"None": TEAL, "Insomnia": PURPLE, "Sleep Apnea": ORANGE},
            symbol="Sleep Disorder",
            symbol_map={"None": "circle", "Insomnia": "triangle-up", "Sleep Apnea": "square"},
            opacity=0.72,
            title="Sleep Quality vs Stress Level — by Disorder",
            labels={"Stress Level": "Stress Level (1–10)", "Quality of Sleep": "Sleep Quality (1–10)"}
        )
    fig.update_traces(marker=dict(size=8))
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, height=400)


# ── 4. Correlation bars ───────────────────────────────────────────────────────
def plot_correlation_bars(df):
    target = "Quality of Sleep"
    numeric_cols = [
        "Sleep Duration", "Stress Level", "Physical Activity Level",
        "Heart Rate", "Daily Steps", "Age"
    ]
    corrs = [(c, df[[c, target]].corr().iloc[0, 1]) for c in numeric_cols if c in df.columns]
    corrs.sort(key=lambda x: x[1])
    labels = [c[0] for c in corrs]
    vals   = [c[1] for c in corrs]
    bar_colors = [TEAL if v > 0 else ORANGE for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="rgba(232,228,248,0.8)")
    ))
    fig.update_xaxes(range=[-1, 1], **GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, "Correlation with Sleep Quality", height=360)


# ── 5. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp   = importances[idx]

    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(sorted_names))]

    fig = go.Figure(go.Bar(
        x=sorted_imp, y=sorted_names, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in sorted_imp],
        textposition="outside",
        textfont=dict(size=10, color="rgba(232,228,248,0.7)")
    ))
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, "Feature Importance — Random Forest", height=360)


# ── 6. Radar chart ────────────────────────────────────────────────────────────
def plot_radar(profiles, n_clusters):
    cats = ["Sleep Duration", "Sleep Quality", "Physical Activity", "Daily Steps", "Low Stress"]
    fig = go.Figure()
    cols = COLORS[:n_clusters]
    for i, p in enumerate(profiles):
        # normalise to 0-100 for radar display
        vals = [
            (p["sleep"] / 9) * 100,
            (p["quality"] / 10) * 100,
            (p["activity"] / 90) * 100,
            (p["steps"] / 10000) * 100,
            ((10 - p["stress"]) / 9) * 100,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            name=f"Cluster {i+1}",
            fill="toself",
            fillcolor=cols[i] + "22",
            line=dict(color=cols[i], width=2),
            marker=dict(color=cols[i], size=6)
        ))
    fig.update_layout(
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
        **LAYOUT,
        title="Cluster Radar Profile",
        height=380
    )
    return fig


# ── 7. Cluster by gender ──────────────────────────────────────────────────────
def plot_cluster_gender(df, n_clusters):
    data = df.groupby(["Gender", "Cluster"]).size().reset_index(name="Count")
    fig = px.bar(
        data, x="Gender", y="Count", color="Cluster",
        barmode="group",
        color_discrete_sequence=COLORS[:n_clusters],
        title="Cluster Distribution by Gender"
    )
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, height=340)


# ── 8. Occupation bars ────────────────────────────────────────────────────────
def plot_occupation_bars(df):
    occ = (
        df.groupby("Occupation")
          .agg(
              Sleep=("Sleep Duration", "mean"),
              Quality=("Quality of Sleep", "mean"),
              Stress=("Stress Level", "mean")
          )
          .reset_index()
          .sort_values("Sleep")
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=occ["Occupation"], x=occ["Sleep"],
        orientation="h", name="Avg Sleep (h)",
        marker=dict(color=PURPLE, line=dict(width=0))
    ))
    fig.add_trace(go.Bar(
        y=occ["Occupation"], x=occ["Quality"],
        orientation="h", name="Avg Quality (/10)",
        marker=dict(color=TEAL, line=dict(width=0))
    ))
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, "Average Sleep Duration & Quality by Occupation", height=400)


# ── 9. BMI vs sleep ───────────────────────────────────────────────────────────
def plot_bmi_sleep(df):
    bmi = (
        df.groupby("BMI Category")
          .agg(
              Sleep=("Sleep Duration", "mean"),
              Quality=("Quality of Sleep", "mean")
          )
          .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bmi["BMI Category"], y=bmi["Sleep"],
        name="Sleep Duration (h)",
        marker=dict(color=TEAL, line=dict(width=0))
    ))
    fig.add_trace(go.Bar(
        x=bmi["BMI Category"], y=bmi["Quality"],
        name="Sleep Quality (/10)",
        marker=dict(color=PURPLE, line=dict(width=0))
    ))
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, "Sleep Metrics by BMI Category", height=340)


# ── 10. Stress & quality by occupation ───────────────────────────────────────
def plot_sleep_by_occupation(df):
    occ = (
        df.groupby("Occupation")
          .agg(
              Stress=("Stress Level", "mean"),
              Quality=("Quality of Sleep", "mean")
          )
          .reset_index()
          .sort_values("Stress", ascending=False)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=occ["Occupation"], y=occ["Stress"],
        name="Avg Stress Level",
        marker=dict(color=ORANGE, line=dict(width=0))
    ))
    fig.add_trace(go.Bar(
        x=occ["Occupation"], y=occ["Quality"],
        name="Avg Sleep Quality",
        marker=dict(color=TEAL, line=dict(width=0))
    ))
    fig.update_xaxes(tickangle=-30, **GRID)
    fig.update_yaxes(**GRID)
    return _apply_layout(fig, "Stress Level vs Sleep Quality by Occupation", height=380)
