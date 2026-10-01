"""Builds the FA2 PDF report (helper script - NOT one of the three files to submit).

Tables come from models/summary.json (written by the notebook); figures are drawn from the dataset and the
saved models, so the report always matches the notebook.

Usage:  python build_report.py --url https://your-app.streamlit.app
"""
import argparse
import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
NAME, PRN = "Shreya Prashant Deshmukh", "125M1H012"
MODELS = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
          "Random Forest": "random_forest", "k-NN": "knn"}
RED, GREEN, NAVY = "#d1495b", "#2a9d8f", "#1d3557"

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="", help="live Streamlit app URL")
parser.add_argument("--out", default=str(HERE / "FA2_Report_Employee_Attrition.pdf"))
args = parser.parse_args()

# DejaVu Sans ships with matplotlib and has the glyphs  ≈ → × – ≥
ttf = Path(font_manager.findfont("DejaVu Sans")).parent
for face, file in {"DV": "DejaVuSans.ttf", "DV-B": "DejaVuSans-Bold.ttf", "DV-I": "DejaVuSans-Oblique.ttf"}.items():
    pdfmetrics.registerFont(TTFont(face, str(ttf / file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

BODY = ParagraphStyle("body", fontName="DV", fontSize=8.8, leading=12.4, spaceAfter=3)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=11, bulletIndent=1, spaceAfter=1.5)
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=12, keepWithNext=1, textColor=colors.HexColor(NAVY), spaceBefore=9, spaceAfter=4)
TITLE = ParagraphStyle("title", fontName="DV-B", fontSize=17, leading=21, alignment=TA_CENTER, textColor=colors.HexColor(NAVY))
SUB = ParagraphStyle("sub", parent=BODY, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("cell", fontName="DV", fontSize=7.4, leading=9.4)
HEADER = ParagraphStyle("hdr", parent=CELL, textColor=colors.white, fontName="DV-B", fontSize=7.0)
CAPTION = ParagraphStyle("cap", parent=BODY, fontName="DV-I", fontSize=7.6, textColor=colors.HexColor("#555555"), alignment=TA_CENTER)

# ---------------------------------------------------------------------------------------------
# Data, saved models, notebook results
# ---------------------------------------------------------------------------------------------
df = pd.read_csv(HERE / "data" / "employee_attrition.csv")
y = (df["Attrition"] == "Yes").astype(int)
X = df.drop(columns=["Attrition", "EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"])
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {n: joblib.load(HERE / "models" / f"{f}.joblib") for n, f in MODELS.items()}
summary = json.loads((HERE / "models" / "summary.json").read_text())
tuning = pd.DataFrame(summary["tuning"]).set_index("Model")
test_tuned = pd.DataFrame(summary["test_tuned"]).set_index("Model")
test_base = pd.DataFrame(summary["test_baseline"]).set_index("Model")
cv_base = pd.DataFrame(summary["cv_baseline"]).set_index("Model")
thr = pd.DataFrame(summary["thresholds"]).set_index("Threshold")
best = tuning["Tuned CV F1"].idxmax()


def fig_image(fig, width_cm):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawHeight, img.drawWidth = width_cm * cm * img.imageHeight / img.imageWidth, width_cm * cm
    return img


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(rows, widths, highlight_row=None):
    cells = [[Paragraph(str(c), HEADER if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(cells, colWidths=[w * cm for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if highlight_row:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#d8f0e6")))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1 - EDA: class balance, attrition rate by overtime / role, income distribution
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [0.8, 1.3, 1.1]})
counts = df["Attrition"].value_counts()
a1.bar(["Stays", "Leaves"], [counts["No"], counts["Yes"]], color=[GREEN, RED])
for i, n in enumerate([counts["No"], counts["Yes"]]):
    a1.text(i, n + 15, f"{n} ({n / len(df):.0%})", ha="center", fontweight="bold")
a1.set(title="Class distribution", ylabel="employees", ylim=(0, counts.max() * 1.15))
rate = df.groupby("JobRole")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).sort_values()
a2.barh(rate.index, rate.values, color=[RED if v > 16.1 else "#e9a3ad" for v in rate.values])
a2.axvline(16.1, color="black", ls="--", lw=0.8)
a2.set(title="Attrition rate by job role (% who left)", xlabel="%  (dashed = overall 16.1 %)")
for cls, name, col in [("No", "Stays", GREEN), ("Yes", "Leaves", RED)]:
    a3.hist(df.loc[df["Attrition"] == cls, "MonthlyIncome"], bins=30, alpha=0.6, color=col, label=name, density=True)
a3.set(title="Monthly income by outcome", xlabel="monthly income", ylabel="density")
a3.legend()
fig.tight_layout()
eda_img = fig_image(fig, 17.2)

# Figure 2 - ROC / PR curves of the tuned models + baseline-vs-tuned F1
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.2))
for name, m in models.items():
    p = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    a1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, p)
    a2.plot(rc, pr, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
a1.plot([0, 1], [0, 1], "k--", alpha=0.35)
a1.set(title="ROC curves (tuned models)", xlabel="false positive rate", ylabel="true positive rate")
a1.legend(fontsize=6.5, loc="lower right")
a2.axhline(y_test.mean(), color="k", ls="--", alpha=0.35)
a2.set(title="Precision-recall curves (dashed = random)", xlabel="recall", ylabel="precision")
a2.legend(fontsize=6.5, loc="upper right")
x = np.arange(len(models))
a3.bar(x - 0.2, test_base.loc[list(models), "F1-Score"], 0.4, color="#adb5bd", label="Baseline")
a3.bar(x + 0.2, test_tuned.loc[list(models), "F1-Score"], 0.4, color=NAVY, label="Tuned")
a3.set(xticks=x, xticklabels=[m.replace(" ", "\n") for m in models], title="Test F1 (leaves class)", ylim=(0, 0.6))
a3.legend()
fig.tight_layout()
perf_img = fig_image(fig, 17.2)

# Figure 3 - threshold trade-off + confusion matrix of the best model at 0.5 vs 0.25
bp = models[best].predict_proba(X_test)[:, 1]
grid = np.linspace(0.05, 0.9, 60)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.0))
a1.plot(grid, [((bp >= t) & (y_test == 1)).sum() / y_test.sum() for t in grid], color=RED, label="Recall")
a1.plot(grid, [(((bp >= t) & (y_test == 1)).sum() / max((bp >= t).sum(), 1)) for t in grid], color=NAVY, label="Precision")
a1.axvline(0.5, color="gray", ls="--", lw=0.8)
a1.set(title=f"{best}: precision / recall vs. decision threshold", xlabel="probability threshold")
a1.legend()
imp = pd.Series(models["Logistic Regression"].named_steps["model"].coef_[0],
                index=[n.split("__", 1)[1] for n in models["Logistic Regression"].named_steps["prep"].get_feature_names_out()])
top = pd.concat([imp.nsmallest(6), imp.nlargest(6)]).sort_values()
a2.barh(top.index, top.values, color=np.where(top.values > 0, RED, GREEN))
a2.set(title="Logistic Regression – strongest coefficients", xlabel="← towards staying | towards leaving →")
fig.tight_layout()
drv_img = fig_image(fig, 17.2)

# ---------------------------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------------------------
link = (f'<b>Live Streamlit app:</b> <a href="{args.url}" color="#1d7a8c"><u>{args.url}</u></a>' if args.url
        else "<b>Live Streamlit app:</b> <font color='#b00020'>[link to be added after deployment]</font>")
S = [Paragraph("Employee Attrition Prediction using Machine Learning", TITLE),
     Paragraph("Formative Assessment-02 · Data Analysis Case Study &amp; Streamlit Deployment", SUB),
     Paragraph("Advanced Data Science [MCA33PE17] · MCA · SYMCA · Academic Year 2026–2027 · Semester I", SUB),
     Paragraph(f"<b>{NAME}</b> · PRN {PRN}", SUB), Spacer(1, 4)]
t = Table([[Paragraph(link, BODY)]], colWidths=[17.4 * cm])
t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1d7a8c")),
                       ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf5f7")), ("TOPPADDING", (0, 0), (-1, -1), 5)]))
S += [t]

S += [Paragraph("1. Problem and dataset", H1),
      Paragraph("<b>Goal:</b> predict whether an employee will <b>leave the company</b> from HR data (role, pay, overtime, tenure, "
                "satisfaction, demographics) so HR can act <i>before</i> a resignation, and compare four classification algorithms. "
                "<b>Dataset:</b> <i>IBM HR Analytics Employee Attrition &amp; Performance</i> – a fictional sample dataset created by IBM data "
                "scientists (Kaggle / IBM GitHub): 1,470 employees, 30 usable features (23 numeric, 7 categorical), "
                "237 leavers (16.1 %) and 1,233 stayers, no missing values.", BODY),
      Paragraph("<b>Why this dataset:</b> a real business problem where the two error types have different costs; mixed numeric and "
                "categorical data that needs genuine preprocessing (one-hot encoding, scaling, dropping constant / ID columns); strong class "
                "imbalance that makes accuracy misleading (always predicting “stays” gives 83.9 %); and a genuinely hard task where model choice, "
                "tuning and honest evaluation matter. Metrics are reported for the <b>leaving</b> class.", BODY)]

S += [Paragraph("2. Exploratory data analysis – key insights", H1), eda_img,
      Paragraph("Figure 1 – class balance, attrition rate by job role, and income distribution of leavers vs. stayers.", CAPTION)]
S += bullets([
    "<b>Clean but imbalanced:</b> no missing values or duplicates; only 16.1 % leave. <font face='DV-I'>EmployeeCount</font>, "
    "<font face='DV-I'>StandardHours</font>, <font face='DV-I'>Over18</font> are constant and <font face='DV-I'>EmployeeNumber</font> is an ID – dropped.",
    "<b>Overtime is the strongest single signal:</b> 30.5 % of employees working overtime left vs. 10.4 % of the others; frequent travellers "
    "24.9 % vs. 8.0 % (non-travel); single employees 25.5 % vs. 10–12.5 %.",
    "<b>Role and seniority:</b> Sales Representatives (39.8 %) and Laboratory Technicians (23.9 %) leave most; Managers (4.9 %) and Research "
    "Directors (2.5 %) least. No stock options: 24.4 %; Job Level 1: 26.3 %.",
    "<b>Leavers are younger, earlier-career and paid less:</b> median income 3,202 vs. 5,204, median age 32 vs. 36, median tenure 3 vs. 6 years.",
    "<b>No numeric feature is decisive:</b> all |r| with attrition ≤ 0.17 (total working years −0.17, job level −0.17, monthly income −0.16) → "
    "a hard problem needing combinations of features. 7 of 253 numeric pairs have |r| &gt; 0.7 (e.g. job level ↔ income). Skewed features contain "
    "genuine (not erroneous) outliers, so they were kept.",
])

S += [Paragraph("3. Preprocessing", H1)]
S += bullets([
    "<b>Columns:</b> 4 useless columns dropped; target encoded Yes → 1, No → 0. <b>Missing values:</b> none; median / most-frequent "
    "<font face='DV-I'>SimpleImputer</font> kept as a safeguard for the deployed app.",
    "<b>Encoding:</b> 7 nominal columns <b>one-hot encoded</b> (51 model features in total) – label encoding would invent a false order between roles. "
    "<b>Scaling:</b> <font face='DV-I'>StandardScaler</font> on numeric features for Logistic Regression and k-NN only (trees are scale-invariant).",
    "<b>No leakage:</b> everything sits in a <font face='DV-I'>ColumnTransformer</font> inside a <font face='DV-I'>Pipeline</font>, re-fitted in every CV fold. "
    "<b>Split:</b> stratified 80 % train (1,176) / 20 % test (294, 47 leavers), seed 42.",
])

rows = [["Model", "Train acc.", "CV accuracy", "CV F1 (leaves)", "CV recall (leaves)"]]
for name, r in cv_base.iterrows():
    rows.append([name, f"{float(r['Train accuracy']):.3f}", r["CV accuracy"], r["CV F1 (leaves)"], r["CV recall (leaves)"]])
S += [Paragraph("4. Models, cross-validation and tuning", H1),
      Paragraph("Four algorithms were implemented with scikit-learn: <b>Logistic Regression, Decision Tree, Random Forest and k-NN</b>. "
                "Baselines were first checked with <b>stratified 5-fold cross-validation</b> on the training data (Table 1): the untuned Decision Tree "
                "and Random Forest reach 100 % training accuracy but only 79 % / 86 % in CV – clear over-fitting. "
                "<b>GridSearchCV</b> (same folds, score = F1 of the leaving class) was then applied to <b>all four</b> models; for LR and the trees the grid also "
                "tested <font face='DV-I'>class_weight='balanced'</font>. The test set was never used to choose parameters.", BODY),
      KeepTogether([table(rows, [3.6, 2.4, 3.6, 3.6, 3.8]), Spacer(1, 2),
                    Paragraph("Table 1 – baseline 5-fold CV results (mean ± std).", CAPTION)])]
rows = [["Model", "Best parameters (GridSearchCV)", "Combos", "Baseline CV F1", "Tuned CV F1"]]
for name, r in tuning.iterrows():
    rows.append([name, r["Best parameters"], r["Combinations"], f"{r['Baseline CV F1']:.4f}", f"{r['Tuned CV F1']:.4f}"])
S += [Spacer(1, 4), KeepTogether([table(rows, [3.2, 6.9, 1.6, 2.8, 2.8]), Spacer(1, 2),
      Paragraph("Table 2 – GridSearchCV: tuning raised CV F1 for the three weaker models; Logistic Regression kept its defaults.", CAPTION)])]

rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Missed leavers", "False alarms", "Baseline F1"]]
order = test_tuned.sort_values(["F1-Score", "ROC-AUC"], ascending=False).index
for name in order:
    r = test_tuned.loc[name]
    rows.append([name, f"{r['Accuracy']:.3f}", f"{r['Precision']:.3f}", f"{r['Recall']:.3f}", f"{r['F1-Score']:.3f}", f"{r['ROC-AUC']:.3f}",
                 f"{r['PR-AUC']:.3f}", int(r["Missed leavers (FN)"]), int(r["False alarms (FP)"]), f"{test_base.loc[name, 'F1-Score']:.3f}"])
S += [Paragraph("5. Results and model comparison (unseen test set, leaves = positive class)", H1),
      KeepTogether([table(rows, [2.7, 1.9, 1.9, 1.3, 1.1, 1.7, 1.5, 1.8, 1.7, 1.7], highlight_row=1), Spacer(1, 2),
                    Paragraph("Table 3 – tuned models on the 294 test employees (47 leavers, default 0.5 threshold); best model highlighted.", CAPTION)]),
      Spacer(1, 4), perf_img, Paragraph("Figure 2 – ROC and precision-recall curves of the tuned models, and baseline-vs-tuned F1.", CAPTION)]

lo = thr.loc[0.25]
S += [Paragraph("6. Insights and conclusions", H1)]
S += bullets([
    f"<b>Best model: {best}</b> – highest CV F1 ({tuning.loc[best, 'Tuned CV F1']:.3f}), ROC-AUC {test_tuned.loc[best, 'ROC-AUC']:.3f} and PR-AUC "
    f"{test_tuned.loc[best, 'PR-AUC']:.3f} (random ≈ 0.16). The signal is mostly additive (overtime, travel, role, income, tenure), which a regularised "
    "linear model captures well from ~1,200 training rows.",
    "<b>Accuracy is misleading:</b> every model reaches 73–86 % accuracy, yet at the default 0.5 threshold the best one finds only 34 % of leavers.",
    "<b>Trees over-fit, tuning helps them most:</b> CV F1 rises 0.356 → 0.429 (Decision Tree), 0.315 → 0.450 (Random Forest), 0.228 → 0.280 (k-NN); "
    "winning grids use <font face='DV-I'>class_weight='balanced'</font> and limited depth. They still do not catch the linear model. k-NN is weakest: "
    "distances over 51 mostly one-hot features are dominated by noise.",
    f"<b>Business view – the threshold matters:</b> lowering the alert threshold from 0.5 to 0.25 raises Logistic Regression's recall from 34 % to "
    f"{lo['Recall']:.0%} (precision {lo['Precision']:.0%}, F1 {lo['F1-Score']:.2f}); HR would talk to {int(lo['Flagged employees'])} of 294 employees "
    f"({lo['Flagged employees'] / 294:.0%}) and catch {int(lo['Leavers caught'])} of 47 leavers.",
    "<b>Drivers:</b> overtime, frequent business travel, Laboratory Technician / Sales Representative roles and long time since promotion raise the risk; "
    "Research Director, no overtime, no travel and more total working years lower it – consistent with the EDA.",
    "<b>Limitations:</b> fictional single-snapshot data, only 47 test leavers (one employee ≈ 2 points of recall), associations rather than causes. "
    "<b>Not for real employment decisions.</b>",
])
S += [Spacer(1, 3), drv_img, Paragraph("Figure 3 – precision / recall vs. decision threshold, and the strongest Logistic Regression coefficients.", CAPTION)]

S += [Paragraph("7. Streamlit application (deployment)", H1),
      Paragraph("<font face='DV-I'>app.py</font> loads the four tuned pipelines saved with <font face='DV-I'>joblib</font> and provides: "
                "(1) an input form with all 30 HR attributes grouped into Personal / Job / Pay &amp; tenure / Satisfaction tabs (sliders, number boxes, "
                "drop-downs) with a <b>Predict</b> button and one-click examples (typical stayer / leaver, random real test records); "
                "(2) <b>every model's probability of leaving</b>, a consensus vote, an adjustable alert threshold and a “Why?” panel listing the factors that raise "
                "or lower the risk; (3) a <b>Model comparison</b> tab (metrics, confusion matrices, tuning and threshold tables computed live on the test split); "
                "(4) a <b>Dataset &amp; insights</b> tab. Run locally with <font face='DV-I'>streamlit run app.py</font>.", BODY),
      Paragraph(link, BODY),
      Paragraph("<i>Educational demo on a fictional dataset – not for real employment decisions.</i>", BODY)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawCentredString(A4[0] / 2, 0.9 * cm, f"FA2 · Advanced Data Science · Employee Attrition Prediction · {NAME} · page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(args.out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                  title="FA2 Report – Employee Attrition Prediction using Machine Learning", author=NAME).build(
    S, onFirstPage=footer, onLaterPages=footer)
print("wrote", args.out)
