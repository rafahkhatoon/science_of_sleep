"""
utils/data_loader.py
Handles loading, cleaning, and preprocessing the Sleep Health & Lifestyle dataset.
"""

import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Embedded sample dataset (subset) ─────────────────────────────────────────
SAMPLE_CSV = """Person ID,Gender,Age,Occupation,Sleep Duration,Quality of Sleep,Physical Activity Level,Stress Level,BMI Category,Blood Pressure,Heart Rate,Daily Steps,Sleep Disorder
1,Male,27,Software Engineer,6.1,6,42,8,Overweight,126/83,77,4200,None
2,Male,28,Doctor,6.2,6,60,8,Normal,125/80,75,10000,None
3,Male,28,Doctor,6.2,6,60,8,Normal,125/80,75,10000,None
4,Male,28,Sales Representative,5.9,4,30,8,Obese,140/90,85,3000,Sleep Apnea
5,Male,28,Sales Representative,5.9,4,30,8,Obese,140/90,85,3000,Sleep Apnea
6,Male,28,Software Engineer,5.9,4,30,8,Obese,140/90,85,3000,Insomnia
7,Female,29,Nurse,6.3,6,40,7,Normal,120/80,70,6800,None
8,Female,29,Nurse,6.3,6,40,7,Normal,120/80,70,6800,None
9,Male,29,Doctor,6.0,6,45,8,Normal,125/80,75,10000,None
10,Male,29,Software Engineer,6.0,6,45,8,Normal,125/80,75,10000,None
11,Female,30,Nurse,6.5,7,42,7,Normal,120/80,70,7000,Insomnia
12,Female,30,Nurse,6.5,7,42,7,Normal,120/80,70,7000,Insomnia
13,Male,30,Doctor,7.0,7,60,6,Normal,120/80,72,8000,None
14,Male,30,Accountant,7.2,7,55,5,Normal,118/76,70,7500,None
15,Female,30,Teacher,7.3,7,50,5,Normal,115/75,68,7000,None
16,Female,31,Nurse,7.0,7,42,7,Normal,120/80,70,7000,Insomnia
17,Male,31,Lawyer,6.0,6,35,8,Overweight,130/85,80,4500,Insomnia
18,Female,31,Teacher,7.5,8,50,4,Normal,115/75,68,8000,None
19,Male,31,Accountant,7.4,7,55,5,Normal,118/76,68,7500,None
20,Female,31,Nurse,7.0,7,42,7,Normal,120/80,70,7000,Insomnia
21,Male,32,Doctor,7.5,8,60,5,Normal,118/75,70,9000,None
22,Female,32,Nurse,6.8,7,42,6,Normal,120/80,70,7500,Insomnia
23,Male,32,Software Engineer,6.0,5,40,8,Overweight,128/84,78,4000,Insomnia
24,Female,32,Teacher,7.8,8,55,4,Normal,112/74,66,8500,None
25,Male,32,Accountant,7.3,7,55,5,Normal,118/76,68,7500,None
26,Female,33,Nurse,7.1,7,45,6,Normal,120/80,70,7500,None
27,Male,33,Doctor,7.8,8,65,4,Normal,115/75,68,9500,None
28,Female,33,Teacher,7.9,8,55,4,Normal,112/74,66,8500,None
29,Male,33,Lawyer,5.9,5,35,8,Overweight,132/87,82,4000,Insomnia
30,Female,33,Accountant,7.5,8,55,4,Normal,115/75,66,8000,None
31,Male,34,Engineer,7.0,7,50,6,Overweight,122/80,74,6000,None
32,Female,34,Nurse,7.2,7,45,6,Normal,120/80,70,7500,Insomnia
33,Male,34,Doctor,8.0,8,65,4,Normal,115/75,68,9500,None
34,Female,34,Sales Representative,5.8,4,25,8,Obese,142/92,88,2800,Sleep Apnea
35,Male,34,Software Engineer,6.2,5,40,8,Overweight,128/84,78,4000,Insomnia
36,Female,34,Teacher,8.0,8,55,3,Normal,110/72,64,9000,None
37,Male,35,Accountant,7.5,8,58,4,Normal,116/76,68,8000,None
38,Female,35,Nurse,7.3,7,45,6,Overweight,122/80,72,7500,Sleep Apnea
39,Male,35,Doctor,8.1,8,65,3,Normal,113/74,66,10000,None
40,Female,35,Teacher,8.1,8,55,3,Normal,110/72,64,9000,None
41,Male,35,Lawyer,6.0,5,35,8,Obese,138/90,84,3500,Insomnia
42,Female,35,Accountant,7.6,8,55,4,Normal,115/75,66,8000,None
43,Male,36,Engineer,7.2,7,50,6,Normal,120/78,72,6500,None
44,Female,36,Nurse,7.4,7,45,6,Overweight,122/80,72,7500,Sleep Apnea
45,Male,36,Doctor,8.1,8,65,3,Normal,113/74,66,10000,None
46,Female,36,Teacher,8.2,8,58,3,Normal,110/72,64,9000,None
47,Male,36,Software Engineer,6.3,5,42,8,Overweight,128/84,78,4200,Insomnia
48,Female,36,Sales Representative,5.9,4,25,9,Obese,142/92,88,2800,Sleep Apnea
49,Male,37,Accountant,7.6,8,58,4,Normal,116/76,68,8000,None
50,Female,37,Nurse,7.5,8,48,5,Normal,118/78,68,8000,None
51,Male,37,Doctor,8.2,8,68,3,Normal,112/73,65,10000,None
52,Female,37,Teacher,8.3,9,58,3,Normal,108/70,62,9500,None
53,Male,37,Lawyer,6.1,5,35,8,Obese,138/90,84,3500,Insomnia
54,Female,37,Accountant,7.7,8,58,4,Normal,115/75,66,8000,None
55,Male,38,Engineer,7.3,7,52,5,Normal,120/78,72,6800,None
56,Female,38,Nurse,7.5,8,48,5,Normal,118/78,68,8000,None
57,Male,38,Doctor,8.3,9,68,3,Normal,112/73,65,10000,None
58,Female,38,Teacher,8.4,9,60,2,Normal,108/70,62,9500,None
59,Male,38,Software Engineer,6.4,5,42,8,Overweight,126/83,76,4500,Insomnia
60,Female,38,Sales Representative,6.0,4,25,9,Obese,142/92,88,2800,Sleep Apnea
61,Male,39,Accountant,7.8,8,60,3,Normal,114/74,66,8500,None
62,Female,39,Nurse,7.6,8,50,5,Normal,118/78,68,8200,None
63,Male,39,Doctor,8.3,9,68,3,Normal,112/73,65,10000,None
64,Female,39,Teacher,8.5,9,60,2,Normal,108/70,62,9500,None
65,Male,39,Lawyer,6.2,5,35,8,Obese,138/90,84,3500,Insomnia
66,Female,39,Accountant,7.8,8,60,3,Normal,114/74,66,8500,None
67,Male,40,Engineer,7.5,8,55,5,Normal,118/76,70,7200,None
68,Female,40,Nurse,7.7,8,50,5,Overweight,120/80,70,8200,Sleep Apnea
69,Male,40,Doctor,8.4,9,70,2,Normal,110/72,63,10000,None
70,Female,40,Teacher,8.5,9,62,2,Normal,107/70,61,9800,None
71,Male,40,Software Engineer,6.5,5,45,7,Overweight,126/83,76,4800,Insomnia
72,Female,40,Sales Representative,6.0,4,25,9,Obese,142/92,88,2800,Sleep Apnea
73,Male,41,Accountant,7.9,8,62,3,Normal,114/74,66,8800,None
74,Female,41,Nurse,7.8,8,52,4,Normal,116/76,67,8500,None
75,Male,41,Doctor,8.5,9,70,2,Normal,110/72,63,10000,None
76,Female,41,Teacher,8.6,9,62,2,Normal,107/70,61,9800,None
77,Male,41,Lawyer,6.3,5,38,8,Obese,138/90,84,3800,Insomnia
78,Female,41,Accountant,7.9,8,62,3,Normal,113/73,65,8800,None
79,Male,42,Engineer,7.6,8,55,4,Normal,117/76,70,7500,None
80,Female,42,Nurse,7.9,8,52,4,Normal,116/76,67,8500,None
81,Male,42,Doctor,8.5,9,70,2,Normal,110/72,63,10000,None
82,Female,42,Teacher,8.6,9,62,2,Normal,107/70,61,9800,None
83,Male,42,Software Engineer,6.5,5,45,7,Overweight,125/82,75,5000,Insomnia
84,Female,42,Sales Representative,6.1,4,28,9,Obese,140/90,86,3000,Sleep Apnea
85,Male,43,Accountant,8.0,8,62,3,Normal,114/74,66,9000,None
86,Female,43,Nurse,8.0,8,54,4,Normal,115/75,66,8800,None
87,Male,43,Doctor,8.5,9,72,2,Normal,110/72,63,10000,None
88,Female,43,Teacher,8.6,9,64,2,Normal,107/70,61,9800,None
89,Male,43,Lawyer,6.3,5,38,8,Obese,137/89,83,3800,Insomnia
90,Female,43,Accountant,8.0,8,62,3,Normal,113/73,65,9000,None
91,Male,44,Engineer,7.7,8,58,4,Normal,116/75,69,7800,None
92,Female,44,Nurse,8.1,8,54,4,Normal,115/75,66,9000,None
93,Male,44,Doctor,8.6,9,72,2,Normal,109/71,62,10000,None
94,Female,44,Teacher,8.7,9,65,2,Normal,106/69,60,9800,None
95,Male,44,Software Engineer,6.6,5,45,7,Overweight,125/82,75,5200,Insomnia
96,Female,44,Sales Representative,6.1,4,28,9,Obese,140/90,86,3000,Sleep Apnea
97,Male,45,Accountant,8.1,8,64,3,Normal,113/73,65,9200,None
98,Female,45,Nurse,8.1,8,55,4,Normal,115/75,66,9000,None
99,Male,45,Manager,7.8,8,60,5,Normal,118/76,70,7800,None
100,Female,45,Manager,7.7,8,58,5,Normal,118/76,70,7500,None
""".strip()


def load_and_preprocess(uploaded_file=None):
    """Load raw CSV, clean it, engineer features, return (raw_df, processed_df)."""

    # ── Load ──────────────────────────────────────────────────────────────────
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_csv(io.StringIO(SAMPLE_CSV))

    df = raw.copy()

    # ── Rename columns for convenience ───────────────────────────────────────
    rename_map = {
        "Sleep Duration (hours)": "Sleep Duration",
        "Quality of Sleep (scale: 1-10)": "Quality of Sleep",
        "Physical Activity Level (minutes/day)": "Physical Activity Level",
        "Stress Level (scale: 1-10)": "Stress Level",
        "Blood Pressure (systolic/diastolic)": "Blood Pressure",
        "Heart Rate (bpm)": "Heart Rate",
    }
    df.rename(columns=rename_map, inplace=True)

    # ── Drop duplicates & missing ─────────────────────────────────────────────
    df.drop_duplicates(subset=["Person ID"], keep="first", inplace=True)
    df.dropna(subset=["Sleep Duration", "Quality of Sleep", "Stress Level"], inplace=True)

    # ── Normalise BMI Category ─────────────────────────────────────────────────
    df["BMI Category"] = df["BMI Category"].str.strip().replace({
        "Normal Weight": "Normal"
    })

    # ── Parse Blood Pressure ───────────────────────────────────────────────────
    if "Blood Pressure" in df.columns:
        try:
            bp_split = df["Blood Pressure"].str.split("/", expand=True)
            df["Systolic BP"] = pd.to_numeric(bp_split[0], errors="coerce")
            df["Diastolic BP"] = pd.to_numeric(bp_split[1], errors="coerce")
        except Exception:
            df["Systolic BP"] = np.nan
            df["Diastolic BP"] = np.nan

    # ── Fill remaining NaNs ────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── Encode categoricals for ML ────────────────────────────────────────────
    le_gender = LabelEncoder()
    le_bmi = LabelEncoder()
    le_disorder = LabelEncoder()

    df["Gender_enc"] = le_gender.fit_transform(df["Gender"])
    df["BMI_enc"] = le_bmi.fit_transform(df["BMI Category"])
    df["Disorder_enc"] = le_disorder.fit_transform(df["Sleep Disorder"])

    # ── Derived features ──────────────────────────────────────────────────────
    df["Sleep Efficiency"] = df["Quality of Sleep"] / df["Sleep Duration"]
    df["Sleep Quality Label"] = df["Quality of Sleep"].apply(
        lambda x: "High Sleep Quality" if x >= 7 else "Low Sleep Quality"
    )

    return raw, df
