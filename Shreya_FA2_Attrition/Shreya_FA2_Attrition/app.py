"""Employee Attrition Predictor - Streamlit app for the FA2 case study.

Loads the four tuned models saved by the notebook (models/*.joblib), lets the user describe an employee
through grouped inputs, and shows every model's probability that the employee will leave, a consensus,
and the factors behind the Logistic Regression prediction.

Run locally:  streamlit run app.py
Educational project - not for real employment decisions.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
DATA_FILE = HERE / "data" / "employee_attrition.csv"
MODEL_FILES = {                       # display name -> file saved by the notebook
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "k-NN": "knn.joblib",
}
RANDOM_STATE = 42                     # must match the notebook so the test split is identical
DROP_COLS = ["Attrition", "EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

GROUPS = {                            # tab name -> inputs shown in it
    "👤 Personal": ["Age", "Gender", "MaritalStatus", "DistanceFromHome", "Education", "EducationField"],
    "💼 Job": ["Department", "JobRole", "JobLevel", "BusinessTravel", "OverTime", "JobInvolvement",
               "StockOptionLevel", "NumCompaniesWorked"],
    "💰 Pay & tenure": ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate", "PercentSalaryHike",
                        "PerformanceRating", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
                        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"],
    "😊 Satisfaction": ["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"],
}
HELP = {                              # tooltips (rating scales follow the dataset documentation)
    "Education": "1 Below College · 2 College · 3 Bachelor · 4 Master · 5 Doctor",
    "EnvironmentSatisfaction": "1 Low · 2 Medium · 3 High · 4 Very High",
    "JobSatisfaction": "1 Low · 2 Medium · 3 High · 4 Very High",
    "RelationshipSatisfaction": "1 Low · 2 Medium · 3 High · 4 Very High",
    "JobInvolvement": "1 Low · 2 Medium · 3 High · 4 Very High",
    "WorkLifeBalance": "1 Bad · 2 Good · 3 Better · 4 Best",
    "PerformanceRating": "Latest rating (3 Excellent · 4 Outstanding in this dataset)",
    "JobLevel": "1 (entry) to 5 (senior)",
    "StockOptionLevel": "0 = no stock options",
    "OverTime": "Does the employee regularly work overtime?",
    "DistanceFromHome": "Distance from home to work (km)",
    "MonthlyIncome": "Monthly income (dataset currency units)",
    "PercentSalaryHike": "Last salary increase, %",
    "YearsSinceLastPromotion": "Years since the last promotion",
}

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")


# ---------------------------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Dataset plus the same stratified 80/20 test split that the notebook used."""
    df = pd.read_csv(DATA_FILE)
    y = (df["Attrition"] == "Yes").astype(int)
    X = df.drop(columns=DROP_COLS)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    return df, X, y, X_test, y_test


@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    info = joblib.load(MODEL_DIR / "feature_info.joblib")
    return models, info


@st.cache_data(show_spinner="Evaluating the models on the held-out test set…")
def test_set_results():
    """Metrics (threshold 0.5) and confusion matrices of the saved models on the unseen test records."""
    models, info = load_models()
    _, _, _, X_test, y_test = load_data()
    rows, matrices = {}, {}
    for name, model in models.items():
        pred = model.predict(X_test[info["features"]])
        p = model.predict_proba(X_test[info["features"]])[:, 1]
        matrices[name] = confusion_matrix(y_test, pred, labels=[0, 1])
        rows[name] = {
            "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred), "F1-Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, p), "PR-AUC": average_precision_score(y_test, p),
            "Missed leavers": int(matrices[name][1, 0]), "False alarms": int(matrices[name][0, 1]),
        }
    return pd.DataFrame(rows).T.astype({"Missed leavers": int, "False alarms": int}), matrices


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def key(feature):
    return f"input::{feature}"


def fill_inputs(values, record=None):
    """Button callback: write feature values into the widget state (runs before the rerun)."""
    for f, v in values.items():
        st.session_state[key(f)] = v.item() if hasattr(v, "item") else v
    st.session_state["record"] = record


def random_record(X_test, y_test, leaver):
    pool = y_test[y_test == int(leaver)].index
    idx = int(np.random.default_rng().choice(pool))
    fill_inputs(X_test.loc[idx], record=idx)


def typical(X, y, leaver):
    """Median (numeric) / most common (categorical) value among leavers or stayers."""
    sub = X[y == int(leaver)]
    return pd.Series({c: (sub[c].mode().iloc[0] if sub[c].dtype == object else int(sub[c].median())) for c in X.columns})


def high_risk(X, y):
    """Illustrative high-risk profile: typical leaver with the strongest risk factors from the EDA."""
    p = typical(X, y, leaver=True)
    p.update({"OverTime": "Yes", "BusinessTravel": "Travel_Frequently", "MaritalStatus": "Single", "Department": "Sales",
              "JobRole": "Sales Representative", "JobLevel": 1, "StockOptionLevel": 0, "Age": 24, "MonthlyIncome": 2300,
              "TotalWorkingYears": 2, "YearsAtCompany": 1, "YearsInCurrentRole": 0, "YearsWithCurrManager": 0,
              "JobSatisfaction": 1, "WorkLifeBalance": 2})
    return p


def pretty(feature):
    return "".join(f" {c}" if c.isupper() and i else c for i, c in enumerate(feature)).strip()


def readable(names, categorical):
    out = []
    for n in names:
        n = n.split("__", 1)[1]
        for c in categorical:
            if n.startswith(c + "_"):
                n = f"{c} = {n[len(c) + 1:]}"
                break
        out.append(n)
    return out


def explain(model, row, categorical):
    """Per-feature contribution to the Logistic Regression log-odds (relative to an average employee)."""
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    x = prep.transform(row)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contrib = pd.Series(x[0] * lr.coef_[0], index=readable(prep.get_feature_names_out(), categorical))
    return contrib[contrib.abs() > 0.05].sort_values()


# ---------------------------------------------------------------------------------------------
# Load everything (friendly error instead of a stack trace if a file is missing)
# ---------------------------------------------------------------------------------------------
try:
    models, info = load_models()
    df, X, y, X_test, y_test = load_data()
except Exception as exc:  # noqa: BLE001 - show any loading problem to the user
    st.error(f"Could not load the trained models or data. Run the notebook first to create them.\n\n`{exc}`")
    st.stop()

features = info["features"]
defaults = typical(X, y, leaver=False)
for f in features:                                         # initial widget values = typical stayer
    st.session_state.setdefault(key(f), defaults[f].item() if hasattr(defaults[f], "item") else defaults[f])
st.session_state.setdefault("record", None)

# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚡ Quick examples")
    st.caption("Fill all 30 inputs in one click, then press **Predict**.")
    st.button("🟢 Typical employee who stayed", width="stretch", on_click=fill_inputs, args=(typical(X, y, False),))
    st.button("🔴 Typical employee who left", width="stretch", on_click=fill_inputs, args=(typical(X, y, True),))
    st.button("🚨 High-risk example profile", width="stretch", on_click=fill_inputs, args=(high_risk(X, y),),
              help="Overtime, frequent travel, single, junior sales role, low pay, short tenure – the strongest risk factors found in the EDA.")
    st.button("🎲 Random real leaver (test set)", width="stretch", on_click=random_record, args=(X_test, y_test, True),
              help="A record the models never saw during training; the recorded outcome is revealed after prediction.")
    st.button("🎲 Random real stayer (test set)", width="stretch", on_click=random_record, args=(X_test, y_test, False))
    st.divider()
    threshold = st.slider("Alert threshold (probability of leaving)", 0.10, 0.90, 0.50, 0.05,
                          help="An employee is flagged 'likely to leave' when a model's probability reaches this value. "
                               "Lower = catch more leavers but more false alarms. The notebook shows 0.25 catches ~62 % "
                               "of leavers (vs 34 % at 0.5).")
    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        "Four classifiers trained on the **IBM HR Analytics Employee Attrition** dataset "
        "(1,470 employees, 30 HR attributes) and tuned with `GridSearchCV`:\n"
        "- Logistic Regression\n- Decision Tree\n- Random Forest\n- k-Nearest Neighbors"
    )
    st.warning("Educational project on a fictional dataset. **Not for real employment decisions.**", icon="⚠️")

# ---------------------------------------------------------------------------------------------
# Header + tabs
# ---------------------------------------------------------------------------------------------
st.title("👥 Employee Attrition Predictor")
st.markdown(
    "Describe an employee and **four machine-learning models** will each estimate the probability that the "
    "employee will **leave the company**."
)

tab_predict, tab_compare, tab_data = st.tabs(["🔬 Predict", "📊 Model comparison", "🗂️ Dataset & insights"])

# ------------------------------- Tab 1: prediction -------------------------------------------
with tab_predict:
    st.markdown("##### 1 · Describe the employee")
    st.caption("Start from a quick example in the sidebar, or change the inputs. Hover the ⓘ icons for details.")

    with st.form("input_form"):
        shown = {g: [f for f in fs if f in features] for g, fs in GROUPS.items()}
        shown["🧩 Other"] = [f for f in features if all(f not in fs for fs in GROUPS.values())]
        shown = {g: fs for g, fs in shown.items() if fs}
        for tab, (group, fs) in zip(st.tabs(list(shown)), shown.items()):
            with tab:
                cols = st.columns(2)
                for i, f in enumerate(fs):
                    with cols[i % 2]:
                        if f in info["categorical"]:
                            st.selectbox(pretty(f), sorted(X[f].unique()), key=key(f), help=HELP.get(f))
                        else:
                            lo, hi = int(X[f].min()), int(X[f].max())
                            widget = st.number_input if hi - lo > 200 else st.slider
                            widget(pretty(f), min_value=lo, max_value=hi, step=1, key=key(f), help=HELP.get(f))
        submitted = st.form_submit_button("🔍 Predict", type="primary", width="stretch")

    if submitted:
        row = pd.DataFrame([{f: st.session_state[key(f)] for f in features}])[features]
        probs = {name: float(m.predict_proba(row)[0, 1]) for name, m in models.items()}
        flagged = {name: p >= threshold for name, p in probs.items()}
        votes, avg_p = sum(flagged.values()), float(np.mean(list(probs.values())))
        leave = votes * 2 > len(probs) or (votes * 2 == len(probs) and avg_p >= threshold)

        st.markdown("##### 2 · Result")
        banner = (f"### {'⚠️' if leave else '✅'} Consensus: **{'Likely to leave' if leave else 'Likely to stay'}** — "
                  f"{votes if leave else len(probs) - votes} of {len(probs)} models agree · average attrition risk **{avg_p:.0%}**")
        (st.warning if leave else st.success)(banner)

        for col, (name, p) in zip(st.columns(len(probs)), probs.items()):
            with col.container(border=True):
                st.markdown(f"**{name}**")
                st.markdown(f"### {'⚠️ Leaves' if flagged[name] else '✅ Stays'}")
                st.progress(min(p, 1.0), text=f"Probability of leaving: {p:.1%}")

        if votes not in (0, len(probs)):
            st.info("The models disagree on this employee – the profile lies near the boundary between the classes.")

        record = st.session_state["record"]
        if record is not None and np.array_equal(row.iloc[0].astype(str).values, X_test.loc[record, features].astype(str).values):
            outcome = "left the company" if y_test.loc[record] == 1 else "stayed"
            st.info(f"This is real test record **#{record}**; in the dataset this employee **{outcome}**.", icon="📋")

        with st.expander("Why? Factors behind the Logistic Regression prediction", expanded=True):
            contrib = explain(models["Logistic Regression"], row, info["categorical"])
            up, down = contrib[contrib > 0].sort_values(ascending=False).head(5), contrib[contrib < 0].head(5)
            c1, c2 = st.columns(2)
            c1.markdown("**⬆️ Increase the risk of leaving**")
            c1.markdown("\n".join(f"- {n}" for n in up.index) or "_none_")
            c2.markdown("**⬇️ Reduce the risk of leaving**")
            c2.markdown("\n".join(f"- {n}" for n in down.index) or "_none_")
            st.caption("Largest contributions to the model's log-odds relative to an average employee profile.")
        st.caption("⚠️ Educational demo – predictions are probabilities from a small, fictional dataset.")

# ------------------------------- Tab 2: model comparison -------------------------------------
with tab_compare:
    metrics, matrices = test_set_results()
    n_pos = int(y_test.sum())
    st.markdown(f"##### Performance on the {len(y_test)} held-out test employees ({n_pos} left) – default 0.5 threshold, *leaves* = positive class")
    float_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    st.dataframe(metrics.style.format("{:.3f}", subset=float_cols).highlight_max(subset=float_cols, color="#b7e4c7"),
                 width="stretch")
    st.caption("Green = best per metric. Only ~16 % of employees leave, so accuracy is misleading – focus on **Recall** "
               "(share of leavers found), **Precision**, **F1** and **PR-AUC**.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Metric comparison")
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        metrics[["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]].plot.bar(ax=ax, rot=15, width=0.8)
        ax.set(ylim=(0, 1.0), ylabel="Score")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("##### Confusion matrices")
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 5))
        for ax, (name, cm) in zip(axes.ravel(), matrices.items()):
            ConfusionMatrixDisplay(cm, display_labels=["Stays", "Leaves"]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    summary_file = MODEL_DIR / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        st.markdown("##### GridSearchCV tuning results (5-fold CV on the training data, score = F1 of the leaving class)")
        st.dataframe(pd.DataFrame(summary["tuning"])[["Model", "Best parameters", "Baseline CV F1", "Tuned CV F1"]],
                     hide_index=True, width="stretch")
        if "thresholds" in summary:
            st.markdown("##### Choosing the alert threshold (tuned Logistic Regression, test set)")
            st.dataframe(pd.DataFrame(summary["thresholds"]).style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}", "Threshold": "{:.2f}"}),
                hide_index=True, width="stretch")

# ------------------------------- Tab 3: dataset ----------------------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", len(df))
    c2.metric("Input features", len(features))
    c3.metric("Left the company", int(y.sum()))
    c4.metric("Attrition rate", f"{y.mean():.1%}")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Attrition rate by overtime and travel (%)")
        for col in ["OverTime", "BusinessTravel", "MaritalStatus"]:
            st.caption(col)
            st.bar_chart((df.groupby(col)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)).sort_values(),
                         horizontal=True, height=140)
    with right:
        st.markdown("##### Attrition rate by job role (%)")
        st.bar_chart(df.groupby("JobRole")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).sort_values(),
                     horizontal=True, height=420)
    with st.expander("Preview the raw data"):
        st.dataframe(df.drop(columns=[c for c in DROP_COLS if c != "Attrition"]).head(15), width="stretch")
    st.caption("Source: IBM HR Analytics Employee Attrition & Performance – fictional sample dataset created by IBM data scientists.")
