import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# -----------------------------------
# Project directories
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "tickets.csv"

CATEGORY_MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"
PRIORITY_MODEL_PATH = BASE_DIR / "models" / "priority_classifier.joblib"


# -----------------------------------
# Load dataset
# -----------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Number of tickets: {len(df)}")
print()


# -----------------------------------
# Remove missing values
# -----------------------------------

df = df.dropna(
    subset=["ticket_text", "category", "priority"]
)


# -----------------------------------
# Input
# -----------------------------------

X = df["ticket_text"].astype(str)


# ==================================================
# MODEL 1: CATEGORY CLASSIFIER
# ==================================================

print("Training category model...")

y_category = df["category"].astype(str)

category_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# Train category model
category_model.fit(X, y_category)


# Save category model
CATEGORY_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    category_model,
    CATEGORY_MODEL_PATH
)

print("Category model trained successfully!")
print(f"Saved at: {CATEGORY_MODEL_PATH}")
print()


# ==================================================
# MODEL 2: PRIORITY CLASSIFIER
# ==================================================

print("Training priority model...")

y_priority = df["priority"].astype(str)

priority_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# Train priority model
priority_model.fit(X, y_priority)


# Save priority model
joblib.dump(
    priority_model,
    PRIORITY_MODEL_PATH
)

print("Priority model trained successfully!")
print(f"Saved at: {PRIORITY_MODEL_PATH}")
print()


# ==================================================
# TEST BOTH MODELS
# ==================================================

sample_ticket = [
    "My order has not arrived and I need an update."
]


category_prediction = category_model.predict(
    sample_ticket
)

priority_prediction = priority_model.predict(
    sample_ticket
)


print("-----------------------------------")
print("TEST PREDICTION")
print("-----------------------------------")

print(
    "Predicted Category:",
    category_prediction[0]
)

print(
    "Predicted Priority:",
    priority_prediction[0]
)

print("-----------------------------------")
print("Training completed successfully!")