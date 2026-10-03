# 🎫 AI Customer Support Ticket Triage

An AI-powered customer support ticket triage system that automatically analyzes customer support tickets, predicts their category and priority, recommends the appropriate support queue, and sends low-confidence predictions for human review.

---

## 📌 Project Overview

Customer support teams receive a large number of support tickets every day. Manually reading, categorizing, prioritizing, and routing every ticket can be time-consuming.

This project uses Machine Learning and Natural Language Processing (NLP) to automate the initial ticket-triage process.

The system accepts a customer's support issue as text and performs the following tasks:

1. Predicts the ticket category.
2. Predicts the ticket priority.
3. Calculates prediction confidence.
4. Recommends the appropriate support queue.
5. Flags low-confidence predictions for human review.
6. Maintains a Human Review Queue.
7. Provides a Streamlit-based dashboard.

---

## 🎯 Objectives

- Automatically classify customer support tickets.
- Identify the priority level of incoming tickets.
- Route tickets to the appropriate support team.
- Reduce manual ticket-triage effort.
- Identify uncertain predictions using confidence scores.
- Send low-confidence tickets to human reviewers.
- Provide an easy-to-use web interface for support teams.

---

## 🧠 Machine Learning Approach

The project uses two separate Machine Learning classification models.

### Category Classification

The category model predicts:

- Delivery
- Payment
- Account
- Return
- Product Issue
- Order Cancellation

### Priority Classification

A separate model predicts:

- High
- Medium
- Low

---

## 🔤 Text Processing

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert customer ticket text into numerical features.

The TF-IDF vectorizer uses:

- Lowercase conversion
- English stop-word removal
- Unigrams and bigrams

Configuration:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)
```

---

## 🤖 Classification Algorithm

The project uses **Logistic Regression** for both category and priority classification.

Configuration:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

Pipeline:

```text
Customer Ticket
       ↓
Text Processing
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression
       ↓
Prediction
       ↓
Category + Priority
```

---

## 📊 Dataset

The project uses a dataset containing **60 customer support tickets**.

| Column | Description |
|---|---|
| ticket_id | Unique ticket identifier |
| customer | Customer name |
| ticket_text | Customer support issue |
| category | Ticket category |
| priority | Ticket priority |

### Category Distribution

| Category | Number of Tickets |
|---|---:|
| Delivery | 10 |
| Payment | 10 |
| Account | 10 |
| Return | 10 |
| Product Issue | 10 |
| Order Cancellation | 10 |
| **Total** | **60** |

### Priority Distribution

| Priority | Number of Tickets |
|---|---:|
| High | 22 |
| Medium | 25 |
| Low | 13 |
| **Total** | **60** |

---

## 📈 Model Evaluation

The project contains:

```text
src/evaluate_model.py
```

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix
- Actual vs Predicted Results
- 5-Fold Cross-Validation

The category and priority models are evaluated separately using stratified train-test splitting and cross-validation.

---

## 🖥️ Streamlit Application

Run the application using:

```bash
streamlit run app.py
```

The application provides:

- Customer ticket submission
- AI category prediction
- AI priority prediction
- Prediction confidence
- Support queue recommendation
- Human review detection
- Human Review Queue
- Dashboard statistics

---

## 📊 Dashboard

The dashboard shows:

- 🎫 Tickets Analyzed
- 🔴 High Priority
- 🟠 Medium Priority
- 🟢 Low Priority
- 👨‍💼 Human Review

---

## 🎫 Ticket Submission

Users can enter a customer name and support issue.

Example:

```text
Customer:
Rahul

Ticket:
My card was charged twice for the same order.
Please fix this immediately.
```

---

## 🤖 AI Classification

The application displays:

### Predicted Category

```text
Payment
```

### Predicted Priority

```text
High
```

The application also displays confidence values for both predictions.

---

## 📌 Recommended Support Queue

| Category | Recommended Queue |
|---|---|
| Delivery | Delivery Support |
| Payment | Billing & Payments |
| Account | Account Support |
| Return | Returns Team |
| Product Issue | Product Support |
| Order Cancellation | Order Management |

---

## 🔎 Prediction Confidence Analysis

The application uses prediction probabilities to calculate confidence.

The system considers the lower confidence value between category and priority predictions.

Human Review threshold:

```text
60%
```

### High Confidence

At least 80%:

```text
High-Confidence Prediction
```

The ticket can be processed automatically.

### Moderate Confidence

Between 60% and 80%:

```text
Moderate Confidence
```

Manual verification is recommended.

### Low Confidence

Below 60%:

```text
Human Review Required
```

The ticket is added to the Human Review Queue.

---

## 👨‍💼 Human Review Queue

Low-confidence tickets are stored in:

```text
logs/review_queue.csv
```

The review queue records:

- Timestamp
- Customer
- Ticket text
- Predicted category
- Category confidence
- Predicted priority
- Priority confidence
- Recommended queue
- Review status

Example status:

```text
Pending Review
```

---

## 🛡️ Duplicate Review Protection

The application prevents the same customer ticket from being added repeatedly to the Human Review Queue while it is still pending review.

If the same ticket is analyzed again, the application displays:

```text
This ticket is already present in the Human Review Queue.
```

---

## 🏗️ Project Structure

```text
AI-Customer-Support-Ticket-Triage/
│
├── data/
│   ├── raw/
│   │   └── tickets.csv
│   └── processed/
│
├── models/
│   ├── ticket_classifier.joblib
│   └── priority_classifier.joblib
│
├── src/
│   ├── train_model.py
│   └── evaluate_model.py
│
├── logs/
│   └── review_queue.csv
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit
- Visual Studio Code
- Git
- GitHub

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Customer-Support-Ticket-Triage
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 — Train the Models

```bash
python src/train_model.py
```

This generates:

```text
models/ticket_classifier.joblib
models/priority_classifier.joblib
```

### Step 2 — Evaluate the Models

```bash
python src/evaluate_model.py
```

### Step 3 — Start the Streamlit Application

```bash
streamlit run app.py
```

---

## 🧪 Example Test Cases

### Delivery / High

```text
My package was marked delivered but I never received it.
Please help immediately.
```

Expected:

```text
Category: Delivery
Priority: High
```

### Payment / High

```text
My card was charged twice for the same order.
Please fix this immediately.
```

Expected:

```text
Category: Payment
Priority: High
```

### Account / Medium

```text
The verification link expired and I need a new one.
```

Expected:

```text
Category: Account
Priority: Medium
```

### Product Issue / Low

```text
The product has a small cosmetic scratch but works normally.
```

Expected:

```text
Category: Product Issue
Priority: Low
```

### Order Cancellation / High

```text
Please cancel my order immediately because I selected the wrong item.
```

Expected:

```text
Category: Order Cancellation
Priority: High
```

---

## 🔄 Complete System Workflow

```text
                    Customer
                       │
                       ▼
                Support Ticket
                       │
                       ▼
                Text Processing
                       │
                       ▼
                TF-IDF Vectorizer
                       │
                       ▼
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Category Model     Priority Model
       Logistic Regression Logistic Regression
              │                 │
              ▼                 ▼
          Category           Priority
              │                 │
              └────────┬────────┘
                       │
                       ▼
              Confidence Analysis
                       │
              ┌────────┴─────────┐
              │                  │
       Confidence ≥ 60%       Confidence < 60%
              │                  │
              ▼                  ▼
       Support Queue       Human Review
              │                  │
              ▼                  ▼
       Ticket Routing      Review Queue
```

---

## 🌟 Key Features

- ✅ Automated ticket classification
- ✅ Six support categories
- ✅ Three priority levels
- ✅ TF-IDF text processing
- ✅ Logistic Regression classification
- ✅ Separate category and priority models
- ✅ Prediction confidence analysis
- ✅ Automatic support queue recommendation
- ✅ Human review for low-confidence predictions
- ✅ Duplicate review protection
- ✅ Human Review Queue
- ✅ Streamlit dashboard
- ✅ Model evaluation
- ✅ Confusion matrix
- ✅ Classification reports
- ✅ 5-fold cross-validation

---

## 🔮 Future Improvements

- Increase the size of the training dataset.
- Use real-world customer support ticket data.
- Experiment with sentence embeddings.
- Use transformer-based NLP models.
- Add an API for integration with support systems.
- Add authentication for support agents.
- Use a database instead of CSV-based review logging.
- Add ticket-resolution prediction.
- Add real-time analytics.
- Add agent feedback for continuous model improvement.
- Deploy the application to a cloud platform.

---

## ⚠️ Limitations

The current system is a prototype trained on a relatively small dataset of 60 tickets.

Therefore, model predictions may not generalize to every type of real-world customer support request.

The Human Review mechanism is included to handle predictions where model confidence is below the configured threshold.

For production use, a substantially larger and representative historical dataset would be required.

---

## 👨‍💻 Project Information

**Project:** AI Customer Support Ticket Triage

**Application:** Streamlit

**Machine Learning:** Scikit-learn

**Text Processing:** TF-IDF

**Classification Algorithm:** Logistic Regression

**Dataset Size:** 60 tickets

**Categories:** 6

**Priority Levels:** 3

**Human Review Threshold:** 60%

---

## 📄 License

This project was developed as an internship capstone project.
