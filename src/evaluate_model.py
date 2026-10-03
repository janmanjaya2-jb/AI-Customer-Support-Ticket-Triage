import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "tickets.csv"


print("=" * 60)
print("AI CUSTOMER SUPPORT TICKET TRIAGE")
print("MODEL EVALUATION")
print("=" * 60)


# ==========================================
# 2. LOAD DATASET
# ==========================================

try:

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

except FileNotFoundError:

    print("\nError: Dataset not found!")
    print("Expected location:", DATA_PATH)

    raise SystemExit(1)


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

required_columns = [
    "ticket_text",
    "category"
]

for column in required_columns:

    if column not in df.columns:

        print(
            f"\nError: Missing required column: {column}"
        )

        raise SystemExit(1)


# Remove missing values
df = df.dropna(
    subset=["ticket_text", "category"]
).copy()


# Convert text columns to strings
df["ticket_text"] = (
    df["ticket_text"]
    .astype(str)
    .str.strip()
)

df["category"] = (
    df["category"]
    .astype(str)
    .str.strip()
)


# Remove empty tickets/categories
df = df[
    (df["ticket_text"] != "") &
    (df["category"] != "")
].copy()


print("\nCleaned dataset shape:", df.shape)


print("\nCategory Distribution:")
print(
    df["category"]
    .value_counts()
)


# ==========================================
# 4. CREATE MODEL
# ==========================================

def create_model():

    model = Pipeline([

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

                class_weight="balanced",

                random_state=42

            )
        )

    ])

    return model


# ==========================================
# 5. STANDARD TRAIN-TEST EVALUATION
# ==========================================

def evaluate_classifier(
    X,
    y,
    target_name
):

    print("\n" + "=" * 60)

    print(
        f"EVALUATING {target_name.upper()} CLASSIFIER"
    )

    print("=" * 60)


    # --------------------------------------
    # Dataset information
    # --------------------------------------

    classes = sorted(
        y.unique()
    )


    if len(classes) < 2:

        print(
            "At least two classes are required."
        )

        return


    print("\nTotal samples:", len(X))

    print(
        "Number of classes:",
        len(classes)
    )

    print(
        "Classes:",
        classes
    )


    # --------------------------------------
    # Train-test split
    # --------------------------------------

    class_counts = y.value_counts()


    test_size = max(
        int(len(X) * 0.25 + 0.5),
        1
    )


    test_size = min(
        test_size,
        len(X) - 2
    )


    can_stratify = (

        class_counts.min() >= 2

        and test_size >= len(classes)

        and
        (len(X) - test_size)
        >= len(classes)

    )


    if can_stratify:

        stratify_value = y

        print(
            "\nUsing stratified train-test split."
        )

    else:

        stratify_value = None

        print(
            "\nUsing regular train-test split."
        )


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=42,

        stratify=stratify_value

    )


    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # --------------------------------------
    # Create model
    # --------------------------------------

    model = create_model()


    # --------------------------------------
    # Train
    # --------------------------------------

    model.fit(
        X_train,
        y_train
    )


    print(
        "\nModel trained successfully!"
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------
    # Metrics
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    print(
        "\n" + "-" * 60
    )

    print(
        "EVALUATION METRICS"
    )

    print(
        "-" * 60
    )


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )


    # --------------------------------------
    # Classification report
    # --------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        "-" * 60
    )


    print(

        classification_report(

            y_test,

            y_pred,

            labels=classes,

            zero_division=0

        )

    )


    # --------------------------------------
    # Confusion matrix
    # --------------------------------------

    print(
        "\nConfusion Matrix:"
    )

    print(
        "-" * 60
    )


    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=classes

    )


    confusion_df = pd.DataFrame(

        cm,

        index=classes,

        columns=classes

    )


    print(
        confusion_df
    )


    # --------------------------------------
    # Actual vs Predicted
    # --------------------------------------

    results = pd.DataFrame({

        "Ticket":
            X_test.values,

        "Actual":
            y_test.values,

        "Predicted":
            y_pred

    })


    print(
        "\nActual vs Predicted Results:"
    )

    print(
        "-" * 60
    )


    print(
        results.to_string(
            index=False
        )
    )


    print(
        "\nEvaluation completed successfully!"
    )


# ==========================================
# 6. CROSS-VALIDATION
# ==========================================

def cross_validate_classifier(

    X,
    y,
    target_name

):

    print(
        "\n" + "=" * 60
    )

    print(
        f"5-FOLD CROSS-VALIDATION: "
        f"{target_name.upper()} CLASSIFIER"
    )

    print(
        "=" * 60
    )


    classes = sorted(
        y.unique()
    )


    if len(classes) < 2:

        print(
            "At least two classes are required."
        )

        return


    # --------------------------------------
    # Create model
    # --------------------------------------

    model = create_model()


    # --------------------------------------
    # Stratified K-Fold
    # --------------------------------------

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42

    )


    # --------------------------------------
    # Evaluation metrics
    # --------------------------------------

    scoring = {

        "accuracy":
            "accuracy",

        "precision":
            "precision_weighted",

        "recall":
            "recall_weighted",

        "f1":
            "f1_weighted"

    }


    # --------------------------------------
    # Cross-validation
    # --------------------------------------

    results = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring=scoring

    )


    # --------------------------------------
    # Display results
    # --------------------------------------

    print(
        "\n5-Fold Cross-Validation Results:"
    )

    print(
        "-" * 60
    )


    print(

        "Accuracy  : "
        f"{results['test_accuracy'].mean():.4f}"

    )


    print(

        "Precision : "
        f"{results['test_precision'].mean():.4f}"

    )


    print(

        "Recall    : "
        f"{results['test_recall'].mean():.4f}"

    )


    print(

        "F1-Score  : "
        f"{results['test_f1'].mean():.4f}"

    )


    # --------------------------------------
    # Fold-by-fold F1
    # --------------------------------------

    print(
        "\nF1-Score for each fold:"
    )


    for i, score in enumerate(

        results["test_f1"],

        start=1

    ):

        print(
            f"Fold {i}: {score:.4f}"
        )


    print(
        "\nCross-validation completed successfully!"
    )


# ==========================================
# 7. CATEGORY CLASSIFIER
# ==========================================

X_category = df[
    "ticket_text"
]

y_category = df[
    "category"
]


evaluate_classifier(

    X_category,

    y_category,

    "Category"

)


cross_validate_classifier(

    X_category,

    y_category,

    "Category"

)


# ==========================================
# 8. PRIORITY CLASSIFIER
# ==========================================

if "priority" in df.columns:


    priority_df = df.dropna(

        subset=[
            "ticket_text",
            "priority"
        ]

    ).copy()


    priority_df["priority"] = (

        priority_df["priority"]

        .astype(str)

        .str.strip()

    )


    priority_df = priority_df[

        priority_df["priority"] != ""

    ]


    if priority_df[
        "priority"
    ].nunique() >= 2:


        X_priority = priority_df[
            "ticket_text"
        ]


        y_priority = priority_df[
            "priority"
        ]


        evaluate_classifier(

            X_priority,

            y_priority,

            "Priority"

        )


        cross_validate_classifier(

            X_priority,

            y_priority,

            "Priority"

        )


    else:

        print(
            "\nPriority evaluation skipped."
        )

        print(
            "At least two priority classes "
            "are required."
        )


else:

    print(
        "\nPriority column not found."
    )

    print(
        "Skipping priority evaluation."
    )


# ==========================================
# 9. FINAL MESSAGE
# ==========================================

print(
    "\n" + "=" * 60
)

print(
    "MODEL EVALUATION PROCESS FINISHED"
)

print(
    "=" * 60
)