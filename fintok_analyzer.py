# FinTokFraud: TikTok Scam Detection Analysis
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score



OUTPUT_DIR = "./newcharts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCAM_COLOR      = "#E05252"   
NON_SCAM_COLOR  = "#52A882"   


plt.rcParams.update({"font.family":       "serif",
                    "font.serif":        ["Georgia", "Times New Roman", "DejaVu Serif"],
                    "axes.spines.top":   False,
                    "axes.spines.right": False,
                    "axes.grid":         True,
                    "grid.alpha":        0.3,
                    "grid.linestyle":    "--",
                    "grid.linewidth":    0.6,
                    "axes.axisbelow":    True,   
                    "figure.facecolor":  "white",
                    "axes.facecolor":    "#FAFAFA",  
                    "axes.labelpad":     8,
                    "xtick.major.pad":   6,
                    "axes.titlesize":    13,
                    "axes.labelsize":    11,
                    "xtick.labelsize":   10,
                    "ytick.labelsize":   10,
                    "legend.fontsize":   10,
                    "figure.dpi":        150,
                    "savefig.dpi":       300,
                    "savefig.bbox":      "tight",
                    "savefig.facecolor": "white"})


# Keywords
LEXICON = {"Scarcity Language": ["limited spots", "limited time", "act now", "don't miss", "last chance",
                                "only a few", "days till", "spots left", "hurry", "selling out",
                                "ends soon", "while supplies last", "today only", "before it's gone",
                                "final chance", "lock in", "kickstart", "looking to buy"],
        "Vague Earnings Claims": ["make money", "earn thousands", "passive income", "financial freedom",
                                "get rich", r"make \\$", r"earn \\$", "income stream", "six figures",
                                "7 figures", "make 6", "unlimited income", "replace your income",
                                "quit your job", "financial independence", "side hustle", "sidehustle",
                                "internet money", "online money", "work from home", "online income",
                                "daily pay", "weekly pay", "be your own boss", "easy money",
                                "income proof", "365k", "digital product", "digital products",
                                "dropshipping", "affiliate marketing"],
        "Recruitment Language": ["dm me", "comment below", "link in bio", "join now", "sign up",
                                "get started", "click link", "message me", r"comment.{0,15}start",
                                "grab your", "send me", "reach out", "apply now", "join my team",
                                "mentorship", "coaching program", "free training", "book a call",
                                "drop a comment", "comment info", "comment yes", "comment interested",
                                "tester", "full guide", "supply today", "top of my page",
                                "website", "how can i get this", "guide", "course"],
        "Supplement Deception": ["weight loss", "metabolism", "detox", "burn fat", "supplement",
                                "lose weight", "gut health", "cortisol", "inflammation",
                                "skincare", "collagen", "probiotic", "natural remedy",
                                "clinically proven", "doctor recommended", "ozempic", "glp",
                                "hormone", "belly fat", "bloat", "cleanse", "plateau",
                                "booster", "fasting", "root", "moto", "viral products",
                                "before after", "before/after", "transformation", "results guaranteed"],
        "Credibility / Proof Language": ["proof", "testimonial", "results", "guaranteed", "verified",
                                "trusted", "secret", "method", "system", "blueprint",
                                "step-by-step", "step by step", "changed my life",
                                "authorized distributor", "stan.store", "creator claims"],
        "Risk-Hiding Language": ["not financial advice", "do your own research", "no risk",
                                "risk free", "safe", "guaranteed results", "works for everyone",
                                "false claim", "misleading", "too good to be true"]}

CATEGORY_WEIGHTS = {"Scarcity Language": 1,
                    "Vague Earnings Claims": 1,
                    "Recruitment Language": 2,
                    "Supplement Deception": 2,
                    "Credibility / Proof Language": 1,
                    "Risk-Hiding Language": 2}

COMMENT_FLAGS = ["scam", "fake", "is this real", "not real", "lying",
                 "does this work", "did anyone try", "waste of money",
                 "too good to be true", "proof", "receipt", "website",
                 "how can i get", "link in bio", "doesn't work"]

CATEGORY_SPECIFIC_KEYWORDS = {"Financial": ["forex", "crypto", "trading", "investment", "invest",
                                             "profit", "portfolio", "signals", "trading group",
                                             "copy my trades", "funded account", "kalshi", "freecash",
                                             "money", "cash", "side hustle", "online business"],
                              "Health": ["ozempic", "detox", "weight loss", "belly fat",
                                         "gut health", "cortisol", "hormone", "cleanse",
                                         "supplement", "natural remedy", "metabolism", "booster",
                                         "root", "moto", "plateau", "fasting", "lbs"],
                              "Lifestyle": ["glow up", "dropshipping", "digital product",
                                            "digital products", "affiliate marketing", "course", "coaching",
                                            "work from home", "laptop lifestyle", "guide", "tester",
                                            "niche", "pricing", "online business"]}


# Loading Data
def load_data():
    financial = pd.read_csv("Financial.csv")
    health    = pd.read_csv("Health.csv")
    lifestyle = pd.read_csv("Lifestyle.csv")

    financial["category"] = "Financial"
    health["category"]    = "Health"
    lifestyle["category"] = "Lifestyle"

    for df in [financial, health, lifestyle]:
        df["Scam_clean"] = df["Scam?"].str.strip().str.lower()

    combined = pd.concat([financial, health, lifestyle], ignore_index=True)
    return combined, financial, health, lifestyle


def combine_text_fields(df):
    df = df.copy()

    for col in ["Caption", "Top Comments", "Primary Hashtag"]:
        if col not in df.columns:
            df[col] = ""

    df["all_text"] = (df["Caption"].fillna("").astype(str) + " " +
                      df["Top Comments"].fillna("").astype(str) + " " +
                      df["Primary Hashtag"].fillna("").astype(str))

    return df


def score_post(caption_text):
    caption_lowercase = str(caption_text).lower()
    category_hit_flags = {}

    for category_name, keyword_list in LEXICON.items():
        category_was_hit = False

        for keyword_pattern in keyword_list:
            if re.search(keyword_pattern, caption_lowercase):
                category_was_hit = True
                break  

        category_hit_flags[category_name] = 1 if category_was_hit else 0

    return category_hit_flags


def comment_suspicion_score(comment_text):
    comment_lowercase = str(comment_text).lower()
    score = 0

    for phrase in COMMENT_FLAGS:
        if phrase in comment_lowercase:
            score += 1

    return score


def category_specific_score(row):
    category = row["category"]
    text = str(row["all_text"]).lower()
    score = 0

    for phrase in CATEGORY_SPECIFIC_KEYWORDS.get(category, []):
        if phrase in text:
            score += 1

    return score


def classify(text, min_score=2):
    hit_flags_per_category = score_post(text)

    total_score = 0
    for category_name, hit in hit_flags_per_category.items():
        if hit == 1:
            total_score += CATEGORY_WEIGHTS.get(category_name, 1)

    if total_score >= min_score:
        return "yes"
    else:
        return "no"


def classify_row(row, min_score=2):
    text = row["all_text"]
    hit_flags_per_category = score_post(text)

    total_score = 0
    for category_name, hit in hit_flags_per_category.items():
        if hit == 1:
            total_score += CATEGORY_WEIGHTS.get(category_name, 1)

    total_score += comment_suspicion_score(row.get("Top Comments", ""))
    total_score += category_specific_score(row)

    if total_score >= min_score:
        return "yes"
    else:
        return "no"


def keyword_hit_rates(list_of_captions):
    total_number_of_captions = len(list_of_captions)
    hit_rate_per_category = {}

    for category_name, keyword_list in LEXICON.items():
        number_of_captions_that_hit_this_category = 0

        for caption in list_of_captions:
            caption_lowercase = str(caption).lower()

            for keyword_pattern in keyword_list:
                if re.search(keyword_pattern, caption_lowercase):
                    number_of_captions_that_hit_this_category += 1
                    break  

        if total_number_of_captions > 0:
            hit_rate_per_category[category_name] = (number_of_captions_that_hit_this_category / total_number_of_captions * 100)
        else:
            hit_rate_per_category[category_name] = 0

    return hit_rate_per_category


# Detects language tricks scammers use to bypass keyword filters:
# hashtag flooding, currency symbols/shorthand, and emoji substitution

EMOJI_RE = re.compile("[""\U0001F600-\U0001F64F"
                      "\U0001F300-\U0001F5FF"
                      "\U0001F680-\U0001F9FF"
                      "]+", flags=re.UNICODE)

def detect_evasion(cap):
    cap = str(cap)
    return {"Hashtag flooding (5+ tags)": cap.count("#") > 4,
            "Currency notation": bool(re.search(r"\$\d+|\d+k|\d+\s*figure", cap, re.I)),
            "Emoji-heavy (4+ clusters)":  len(EMOJI_RE.findall(cap)) > 3}


# Generating Charts
def fig1_scam_rate_by_category(combined):
    categories = ["Financial", "Health", "Lifestyle"]
    scam_counts  = []
    legit_counts = []
    scam_rates   = []

    for cat in categories:
        sub = combined[combined["category"] == cat]
        n_scam  = (sub["Scam_clean"] == "yes").sum()
        n_legit = (sub["Scam_clean"] == "no").sum()
        scam_counts.append(n_scam)
        legit_counts.append(n_legit)
        scam_rates.append(n_scam / len(sub) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(categories))
    bars_scam  = ax.bar(x, scam_counts,  label="Scam",       color=SCAM_COLOR,  alpha=0.88)
    bars_legit = ax.bar(x, legit_counts, label="Legitimate",  color=NON_SCAM_COLOR, alpha=0.88,
                        bottom=scam_counts)

    for i, (sc, rate) in enumerate(zip(scam_counts, scam_rates)):
        ax.text(i, sc / 2, f"{rate:.0f}%\nscam", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Number of posts")
    ax.set_title("Scam vs. Legitimate posts (By category)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 150)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig1_scam_rate_by_category.png"
    plt.savefig(path)
    plt.close()


def fig2_keyword_category_hits(combined: pd.DataFrame):
    text_column = "all_text" if "all_text" in combined.columns else "Caption"
    scam_caps  = combined[combined["Scam_clean"] == "yes"][text_column].dropna()
    legit_caps = combined[combined["Scam_clean"] == "no"][text_column].dropna()

    scam_rates  = keyword_hit_rates(scam_caps)
    legit_rates = keyword_hit_rates(legit_caps)

    cats   = list(LEXICON.keys())
    s_vals = [scam_rates[c]  for c in cats]
    l_vals = [legit_rates[c] for c in cats]

    x     = range(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.8))

    b1 = ax.bar([i - width/2 for i in x], s_vals, width, label="Scam posts",
                color=SCAM_COLOR, alpha=0.88)
    b2 = ax.bar([i + width/2 for i in x], l_vals, width, label="Legitimate posts",
                color=NON_SCAM_COLOR, alpha=0.88)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    short_labels = ["Scarcity\nlanguage", "Vague\nearnings", "Recruitment\nlanguage",
                    "Supplement\ndeception", "Credibility/\nproof", "Risk-hiding\nlanguage"]
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Posts containing category keyword (%)")
    ax.set_title("Keyword category hit rates: scam vs. legitimate posts")
    ax.legend(frameon=False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_keyword_category_hits.png"
    plt.savefig(path)
    plt.close()

def fig3_confusion_matrix(combined: pd.DataFrame, min_score=2):
    combined = combined.copy()
    combined["predicted"] = combined.apply(lambda row: classify_row(row, min_score=min_score), axis=1)
    y_true = (combined["Scam_clean"] == "yes").astype(int)
    y_pred = (combined["predicted"]  == "yes").astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cell_labels = np.array([
        [f"True Negative\n{cm[0,0]}",  f"False Positive\n{cm[0,1]}"],
        [f"False Negative\n{cm[1,0]}", f"True Positive\n{cm[1,1]}"]])

    color_matrix = np.array([[0, 1], [1, 0]], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(color_matrix,
                annot=cell_labels, fmt="",
                cmap=LinearSegmentedColormap.from_list("cmap", ["#52A882", "#E05252"]),
                vmin=0, vmax=1,
                xticklabels=["Predicted: Legit", "Predicted: Scam"],
                yticklabels=["Actual: Legit",    "Actual: Scam"],
                ax=ax, linewidths=3, linecolor="white",
                cbar=False, annot_kws={"size": 13, "color": "white", "fontweight": "bold"})

    ax.set_title(f"Keyword classifier — confusion matrix\n(weighted score ≥{min_score}, n = 400)", fontsize=11, pad=14, color="#2D2D2D")
    ax.tick_params(length=0, labelsize=10)

    fig.text(0.5, 0.01, f"Precision: {p:.2f}   |   Recall: {r:.2f}   |   F1: {f1:.2f}",
        ha="center", fontsize=9, color="#555555", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTPUT_DIR}/fig3_confusion_matrix.png"
    plt.savefig(path)
    plt.close()


def fig4_evasion_tactics(combined: pd.DataFrame):
    text_column = "all_text" if "all_text" in combined.columns else "Caption"
    scam_posts  = combined[combined["Scam_clean"] == "yes"][text_column].dropna()
    legit_posts = combined[combined["Scam_clean"] == "no"][text_column].dropna()

    tactic_names = list(detect_evasion("dummy").keys())

    def tactic_rates(posts):
        counts = Counter()
        for cap in posts:
            for k, v in detect_evasion(cap).items():
                if v:
                    counts[k] += 1
        return [counts[t] / len(posts) * 100 for t in tactic_names]

    s_rates = tactic_rates(scam_posts)
    l_rates = tactic_rates(legit_posts)

    x     = range(len(tactic_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    b1 = ax.bar([i - width/2 for i in x], s_rates, width, label="Scam posts",
                color=SCAM_COLOR, alpha=0.88)
    b2 = ax.bar([i + width/2 for i in x], l_rates, width, label="Legitimate posts",
                color=NON_SCAM_COLOR, alpha=0.88)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tactic_names, wrap=True)
    ax.set_ylabel("Posts exhibiting tactic (%)")
    ax.set_title("Evasion tactic prevalence: scam vs. legitimate posts")
    ax.legend(frameon=False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig4_evasion_tactics.png"
    plt.savefig(path)
    plt.close()


def test_thresholds(combined: pd.DataFrame):
    print("\nTHRESHOLD TESTING")
    print("Score | Precision | Recall | F1")

    y_true = (combined["Scam_clean"] == "yes").astype(int)

    for score in [1, 2, 3, 4, 5, 6]:
        preds = combined.apply(lambda row: classify_row(row, min_score=score), axis=1)
        y_pred = (preds == "yes").astype(int)

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{score:5d} | {p:.3f}     | {r:.3f}  | {f1:.3f}")


def show_errors(combined: pd.DataFrame, min_score=2):
    combined = combined.copy()
    combined["predicted"] = combined.apply(lambda row: classify_row(row, min_score=min_score), axis=1)

    false_negatives = combined[(combined["Scam_clean"] == "yes") & (combined["predicted"] == "no")]
    false_positives = combined[(combined["Scam_clean"] == "no") & (combined["predicted"] == "yes")]

    false_negatives.to_csv("false_negatives_to_review.csv", index=False)
    false_positives.to_csv("false_positives_to_review.csv", index=False)

    print(f"\nSaved {len(false_negatives)} false negatives to review.")
    print(f"Saved {len(false_positives)} false positives to review.")


def run_tfidf_logistic_regression(combined: pd.DataFrame):
    X = combined["all_text"].fillna("")
    y = (combined["Scam_clean"] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42, stratify=y)

    model = Pipeline([("tfidf", TfidfVectorizer(lowercase=True,
                                                ngram_range=(1, 2),
                                                min_df=2,
                                                max_df=0.90)),
                      ("logreg", LogisticRegression(max_iter=1000,
                                                    class_weight="balanced"))])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nTF-IDF + Logistic Regression Results")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Scam"], zero_division=0))


import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_kappa_from_file():
    df = pd.read_csv("Cohen's Kappa.csv")  
    df["Scam_clean"] = df["Scam?"].astype(str).str.strip().str.lower()
    df["Second_clean"] = df["Second Label"].astype(str).str.strip().str.lower()

    df = df[df["Scam_clean"].isin(["yes", "no"]) & df["Second_clean"].isin(["yes", "no"])]

    kappa = cohen_kappa_score(df["Scam_clean"], df["Second_clean"])

    print("\nCOHEN'S KAPPA RESULTS")
    print(f"Number of posts: {len(df)}")
    print(f"Cohen's kappa: {kappa:.3f}")

    disagreements = df[df["Scam_clean"] != df["Second_clean"]]
    print(f"Disagreements: {len(disagreements)}")

    disagreements.to_csv("kappa_disagreements.csv", index=False)
    print("Saved disagreements to kappa_disagreements.csv")


def print_summary(combined: pd.DataFrame, min_score=2):
    print("FINTOKFRAUD — SUMMARY STATISTICS\n")

    total = len(combined)
    n_scam  = (combined["Scam_clean"] == "yes").sum()
    n_legit = (combined["Scam_clean"] == "no").sum()
    print(f"\nTotal posts:      {total}")
    print(f"Scam posts:       {n_scam}  ({n_scam/total*100:.1f}%)")
    print(f"Legitimate posts: {n_legit} ({n_legit/total*100:.1f}%)\n")

    for cat in ["Financial", "Health", "Lifestyle"]:
        sub = combined[combined["category"] == cat]
        rate = (sub["Scam_clean"] == "yes").sum() / len(sub) * 100
        print(f"  {cat:12s} scam rate: {rate:.1f}%  (n={len(sub)})")

    # Classifier metrics
    combined = combined.copy()
    combined["predicted"] = combined.apply(lambda row: classify_row(row, min_score=min_score), axis=1)
    y_true = (combined["Scam_clean"] == "yes").astype(int)
    y_pred = (combined["predicted"]  == "yes").astype(int)

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nKeyword classifier (weighted score ≥{min_score}):")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"\nConfusion matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("\n" + "="*60)



def main():
    print("Loading data...")
    combined, financial, health, lifestyle = load_data()
    combined = combine_text_fields(combined)

    chosen_score = 2

    print_summary(combined, min_score=chosen_score)
    test_thresholds(combined)
    show_errors(combined, min_score=chosen_score)
    run_tfidf_logistic_regression(combined)
    compute_kappa_from_file()

    print("\nGenerating charts...")
    fig1_scam_rate_by_category(combined)
    fig2_keyword_category_hits(combined)
    fig3_confusion_matrix(combined, min_score=chosen_score)
    fig4_evasion_tactics(combined)

    print(f"\nDone!")

main()

