"""
Email/SMS Spam Classification - Model Training Script
Task 1: ML Internship - Arch Technologies

Pipeline: Data Cleaning -> Text Preprocessing -> TF-IDF Feature Extraction
          -> Train multiple models -> Compare -> Save best model
"""

import re
import string
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("Step 1: Loading dataset...")
df = pd.read_csv("/mnt/user-data/uploads/spam.csv", encoding="latin-1")

# Keep only the relevant columns and rename
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Drop duplicates and nulls
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print(f"Dataset shape after cleaning: {df.shape}")
print(df["label"].value_counts())

# -----------------------------
# 2. TEXT PREPROCESSING
# -----------------------------
print("\nStep 2: Preprocessing text...")

# Basic English stopwords list (avoids needing nltk download / internet access)
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves
""".split())


def clean_text(text):
    """Lowercase, remove punctuation/numbers/extra whitespace, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                # remove extra whitespace
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


df["clean_message"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print("Sample before/after cleaning:")
print(df[["message", "clean_message"]].head(3).to_string())

# -----------------------------
# 3. TRAIN/TEST SPLIT
# -----------------------------
X = df["clean_message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# -----------------------------
# 4. FEATURE EXTRACTION (TF-IDF)
# -----------------------------
print("\nStep 3: TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 5. TRAIN & COMPARE MULTIPLE MODELS
# -----------------------------
print("\nStep 4: Training and comparing models...")

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1
    })
    trained_models[name] = model
    print(f"{name:30s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# -----------------------------
# 6. SELECT BEST MODEL
# -----------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

best_preds = best_model.predict(X_test_tfidf)
print("\nClassification Report (Best Model):")
print(classification_report(y_test, best_preds, target_names=["Ham", "Spam"]))

# -----------------------------
# 7. CONFUSION MATRIX PLOT
# -----------------------------
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("/home/claude/confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved as confusion_matrix.png")

# -----------------------------
# 8. MODEL COMPARISON BAR CHART
# -----------------------------
plt.figure(figsize=(8, 5))
results_df_plot = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
results_df_plot.plot(kind="bar", figsize=(9, 5))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0.8, 1.0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("/home/claude/model_comparison.png", dpi=150)
print("Model comparison chart saved as model_comparison.png")

# -----------------------------
# 9. SAVE MODEL + VECTORIZER + METADATA
# -----------------------------
joblib.dump(best_model, "/home/claude/spam_model.pkl")
joblib.dump(vectorizer, "/home/claude/tfidf_vectorizer.pkl")
results_df.to_csv("/home/claude/model_results.csv", index=False)

print("\nSaved: spam_model.pkl, tfidf_vectorizer.pkl, model_results.csv")
print("\nDONE.")
