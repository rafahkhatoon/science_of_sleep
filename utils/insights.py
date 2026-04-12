"""
utils/insights.py
Generates data-driven insights from the processed DataFrame.
"""

import pandas as pd
import numpy as np


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """
    Compute real statistics from df and return a list of insight dicts:
    { title, body, icon, color, tag }
    """
    insights = []

    # ── 1. Stress vs Quality ──────────────────────────────────────────────────
    corr_stress = df["Stress Level"].corr(df["Quality of Sleep"])
    insights.append({
        "icon": "🔥",
        "color": "#ff8c69",
        "title": "Stress is the #1 enemy of sleep quality",
        "body": (
            f"Stress level shows a Pearson correlation of <b>{corr_stress:.2f}</b> with sleep quality — "
            "the strongest predictor in this dataset. Subjects with stress ≥ 8 average a sleep quality of "
            f"{df[df['Stress Level'] >= 8]['Quality of Sleep'].mean():.1f}/10, compared to "
            f"{df[df['Stress Level'] <= 3]['Quality of Sleep'].mean():.1f}/10 for those with stress ≤ 3."
        ),
        "tag": "HIGH IMPACT · Correlation Analysis"
    })

    # ── 2. Physical activity ──────────────────────────────────────────────────
    high_act = df[df["Physical Activity Level"] >= 60]["Quality of Sleep"].mean()
    low_act  = df[df["Physical Activity Level"] < 30]["Quality of Sleep"].mean()
    insights.append({
        "icon": "🌿",
        "color": "#4ecdc4",
        "title": "Regular physical activity meaningfully improves sleep",
        "body": (
            f"Subjects exercising ≥60 min/day average a sleep quality of <b>{high_act:.1f}/10</b>, "
            f"versus <b>{low_act:.1f}/10</b> for those exercising less than 30 min/day. "
            f"The correlation between activity and sleep quality is "
            f"{df['Physical Activity Level'].corr(df['Quality of Sleep']):.2f}. "
            "Even a moderate increase in daily movement appears beneficial."
        ),
        "tag": "HIGH IMPACT · Activity Analysis"
    })

    # ── 3. BMI & disorders ────────────────────────────────────────────────────
    if "Sleep Disorder" in df.columns and "BMI Category" in df.columns:
        obese_apnea = (
            df[df["BMI Category"] == "Obese"]["Sleep Disorder"]
            .value_counts(normalize=True)
            .get("Sleep Apnea", 0) * 100
        )
        normal_apnea = (
            df[df["BMI Category"] == "Normal"]["Sleep Disorder"]
            .value_counts(normalize=True)
            .get("Sleep Apnea", 0) * 100
        )
        insights.append({
            "icon": "🩺",
            "color": "#7c6fe0",
            "title": "Obesity is strongly linked to sleep apnea prevalence",
            "body": (
                f"Sleep apnea affects <b>{obese_apnea:.0f}%</b> of obese subjects in this dataset, "
                f"compared to just <b>{normal_apnea:.0f}%</b> of those with normal BMI. "
                "Heart rate and blood pressure readings are also notably elevated in the obese group. "
                "Weight management interventions may have significant downstream benefits for sleep health."
            ),
            "tag": "MEDIUM IMPACT · BMI Analysis"
        })

    # ── 4. Occupation spotlight ───────────────────────────────────────────────
    if "Occupation" in df.columns:
        occ_stress = df.groupby("Occupation")["Stress Level"].mean().sort_values(ascending=False)
        worst_occ  = occ_stress.index[0]
        best_occ   = occ_stress.index[-1]
        insights.append({
            "icon": "💼",
            "color": "#ffd166",
            "title": f"{worst_occ}s report highest stress; {best_occ}s report the lowest",
            "body": (
                f"<b>{worst_occ}s</b> have the highest average stress ({occ_stress.iloc[0]:.1f}/10) "
                f"and correspondingly poor sleep quality "
                f"({df[df['Occupation'] == worst_occ]['Quality of Sleep'].mean():.1f}/10). "
                f"In contrast, <b>{best_occ}s</b> average only {occ_stress.iloc[-1]:.1f}/10 stress "
                f"and {df[df['Occupation'] == best_occ]['Quality of Sleep'].mean():.1f}/10 sleep quality. "
                "Workplace stress management programmes should target high-stress professions."
            ),
            "tag": "MEDIUM IMPACT · Occupation Analysis"
        })

    # ── 5. Sleep duration sweet spot ─────────────────────────────────────────
    dur_quality = df.groupby(df["Sleep Duration"].round(0))["Quality of Sleep"].mean()
    best_dur = dur_quality.idxmax()
    insights.append({
        "icon": "⏰",
        "color": "#4ecdc4",
        "title": f"The sleep quality sweet spot is around {best_dur:.0f} hours",
        "body": (
            f"Sleep quality peaks at approximately <b>{best_dur:.0f} hours</b> of sleep per night "
            f"(avg quality: {dur_quality.max():.1f}/10 in this dataset). "
            f"Only {(df['Sleep Duration'] >= 7).mean()*100:.0f}% of subjects achieve at least 7 hours. "
            "Subjects sleeping under 6 hours average a quality score of "
            f"{df[df['Sleep Duration'] < 6]['Quality of Sleep'].mean():.1f}/10 — well below the dataset mean."
        ),
        "tag": "HIGH IMPACT · Duration Analysis"
    })

    # ── 6. Gender differences ─────────────────────────────────────────────────
    if "Gender" in df.columns:
        g = df.groupby("Gender")[["Quality of Sleep", "Stress Level", "Sleep Duration"]].mean()
        if len(g) >= 2:
            genders = g.index.tolist()
            insights.append({
                "icon": "👥",
                "color": "#a8dadc",
                "title": "Gender differences in sleep patterns are modest but present",
                "body": (
                    f"<b>{genders[0]}s</b> in this dataset average "
                    f"{g.loc[genders[0], 'Sleep Duration']:.1f}h sleep / "
                    f"{g.loc[genders[0], 'Quality of Sleep']:.1f} quality / "
                    f"{g.loc[genders[0], 'Stress Level']:.1f} stress. "
                    f"<b>{genders[1]}s</b> average "
                    f"{g.loc[genders[1], 'Sleep Duration']:.1f}h / "
                    f"{g.loc[genders[1], 'Quality of Sleep']:.1f} quality / "
                    f"{g.loc[genders[1], 'Stress Level']:.1f} stress. "
                    "Disorder prevalence varies more sharply by BMI category than by gender."
                ),
                "tag": "LOW IMPACT · Demographics"
            })

    return insights
