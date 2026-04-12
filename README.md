#  Science of Sleep — Data Mining Dashboard

A full-stack **Streamlit** dashboard for analysing sleep health data using K-Means clustering, correlation analysis, and a Random Forest classifier.

---

##  Project Structure

```
sleep_project/
├── app.py                  ← Main Streamlit entry point
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      ← Load & preprocess CSV
│   ├── charts.py           ← All Plotly chart functions
│   ├── clustering.py       ← K-Means logic
│   ├── ml_models.py        ← Random Forest classifier
│   └── insights.py         ← Auto-generated data insights
└── README.md
```

---

##  Setup

### 1. Clone / copy project files
```bash
cd sleep_project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

##  Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

##  Dataset

Download the **Sleep Health and Lifestyle Dataset** from Kaggle:
> https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

Save the CSV and upload it via the sidebar **"Upload your dataset"** button.

The app ships with a built-in 100-row sample so you can explore without uploading anything.

---

##  Features

| Tab | What it does |
|-----|-------------|
| **Overview** | EDA — histograms, donut charts, scatter plots |
| **Clustering** | K-Means (k=2–5) with radar & gender charts |
| **Correlations** | Pearson bars, heatmap, BMI comparison |
| **By Occupation** | Sleep, quality, stress & disorder by profession |
| **ML Model** | Random Forest — accuracy, confusion matrix, feature importance, live prediction widget |
| **Insights** | 6 auto-computed data-driven findings |

---

##  Tech Stack

- **Streamlit** — web app framework
- **Plotly** — all interactive visualisations
- **scikit-learn** — K-Means clustering + Random Forest
- **Pandas / NumPy** — data manipulation

---

##  Notes

- The sidebar lets you filter by gender and age range in real time
- The ML tab includes a live prediction widget — enter your own stats and get a sleep quality prediction
- The K-Means cluster count is adjustable (2–5) via the sidebar slider
