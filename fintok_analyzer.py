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


OUTPUT_DIR = "./charts"
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
                                "ends soon", "while supplies last"],
        "Vague Earnings Claims": ["make money", "earn thousands", "passive income", "financial freedom",
                                "get rich", r"make \$", r"earn \$", "income stream", "six figures",
                                "7 figures", "make 6", "unlimited income", "replace your income",
                                "quit your job", "financial independence"],
        "Recruitment Language": ["dm me", "comment below", "link in bio", "join now", "sign up",
                                "get started", "click link", "message me", r"comment.{0,10}start",
                                "grab your", "send me", "reach out"],
        "Supplement Deception": ["weight loss", "metabolism", "detox", "burn fat", "supplement",
                                "lose weight", "gut health", "cortisol", "inflammation",
                                "skincare", "collagen", "probiotic", "natural remedy",
                                "clinically proven", "doctor recommended"]}


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



def score_post(caption_text):
    """
    For a single post's caption, checks whether it contains any keywords from each of the four scam lexicon categories

    Returns a dictionary like:
        {"Scarcity Language":       1,   
        "Vague Earnings Claims":   0,   
        "Recruitment Language":    1,
        "Supplement Deception":    0}
    """

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


def classify(caption_text, min_categories_hit=1):
    """
    Decide whether a post is a scam based on how many lexicon categories it triggers

    A post is flagged as a scam if it hits keywords from at least 'min_categories_hit' different categories.

    - min_categories_hit = 1 (default): flag if ANY category matched
    - min_categories_hit = 2: flag only if TWO or more categories matched (stricter, fewer false positives but also fewer true positives)
    """
    hit_flags_per_category = score_post(caption_text)

    # Count how many categories had at least one keyword match
    number_of_categories_hit = sum(hit_flags_per_category.values())

    if number_of_categories_hit >= min_categories_hit:
        return "yes"
    else:
        return "no"


def keyword_hit_rates(list_of_captions):
    """
    Given a list/series of captions, calculate what percentage of them contain at least one keyword from each lexicon category
    """

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
    scam_caps  = combined[combined["Scam_clean"] == "yes"]["Caption"].dropna()
    legit_caps = combined[combined["Scam_clean"] == "no"]["Caption"].dropna()

    scam_rates  = keyword_hit_rates(scam_caps)
    legit_rates = keyword_hit_rates(legit_caps)

    cats   = list(LEXICON.keys())
    s_vals = [scam_rates[c]  for c in cats]
    l_vals = [legit_rates[c] for c in cats]

    x     = range(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    b1 = ax.bar([i - width/2 for i in x], s_vals, width, label="Scam posts",
                color=SCAM_COLOR, alpha=0.88)
    b2 = ax.bar([i + width/2 for i in x], l_vals, width, label="Legitimate posts",
                color=NON_SCAM_COLOR, alpha=0.88)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    short_labels = ["Scarcity\nlanguage", "Vague\nearnings", "Recruitment\nlanguage", "Supplement\ndeception"]
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Posts containing category keyword (%)")
    ax.set_title("Keyword category hit rates: scam vs. legitimate posts")
    ax.legend(frameon=False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_keyword_category_hits.png"
    plt.savefig(path)
    plt.close()

def fig3_confusion_matrix(combined: pd.DataFrame):
    combined = combined.copy()
    combined["predicted"] = combined["Caption"].apply(lambda x: classify(x, min_categories_hit=1))
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

    sns.heatmap(
        color_matrix,
        annot=cell_labels, fmt="",
        cmap=LinearSegmentedColormap.from_list("cmap", ["#52A882", "#E05252"]),
        vmin=0, vmax=1,
        xticklabels=["Predicted: Legit", "Predicted: Scam"],
        yticklabels=["Actual: Legit",    "Actual: Scam"],
        ax=ax, linewidths=3, linecolor="white",
        cbar=False, annot_kws={"size": 13, "color": "white", "fontweight": "bold"})

    ax.set_title(
        "Keyword classifier — confusion matrix\n(threshold ≥1 category, n = 300)",
        fontsize=11, pad=14, color="#2D2D2D")
    ax.tick_params(length=0, labelsize=10)

    fig.text(
        0.5, 0.01,
        f"Precision: {p:.2f}   |   Recall: {r:.2f}   |   F1: {f1:.2f}",
        ha="center", fontsize=9, color="#555555", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTPUT_DIR}/fig3_confusion_matrix.png"
    plt.savefig(path)
    plt.close()


def fig4_evasion_tactics(combined: pd.DataFrame):
    scam_posts  = combined[combined["Scam_clean"] == "yes"]["Caption"].dropna()
    legit_posts = combined[combined["Scam_clean"] == "no"]["Caption"].dropna()

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



def print_summary(combined: pd.DataFrame):
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
    combined["predicted"] = combined["Caption"].apply(lambda x: classify(x, min_categories_hit=1))
    y_true = (combined["Scam_clean"] == "yes").astype(int)
    y_pred = (combined["predicted"]  == "yes").astype(int)

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nKeyword classifier (threshold ≥1 category):")
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

    print_summary(combined)

    print("\nGenerating charts...")
    fig1_scam_rate_by_category(combined)
    fig2_keyword_category_hits(combined)
    fig3_confusion_matrix(combined)
    fig4_evasion_tactics(combined)

    print(f"\nDone!")

main()
