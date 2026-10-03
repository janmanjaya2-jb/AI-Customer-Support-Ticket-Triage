import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CATEGORY_MODEL_PATH = (
    BASE_DIR / "models" / "ticket_classifier.joblib"
)

PRIORITY_MODEL_PATH = (
    BASE_DIR / "models" / "priority_classifier.joblib"
)

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REVIEW_LOG_PATH = (
    LOG_DIR / "review_queue.csv"
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🎫",
    layout="wide"
)


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    category_model = joblib.load(
        CATEGORY_MODEL_PATH
    )

    priority_model = joblib.load(
        PRIORITY_MODEL_PATH
    )

    return (
        category_model,
        priority_model
    )


category_model, priority_model = load_models()


# ============================================================
# 4. SESSION STATE
# ============================================================

if "tickets_analyzed" not in st.session_state:
    st.session_state.tickets_analyzed = 0

if "high_priority" not in st.session_state:
    st.session_state.high_priority = 0

if "medium_priority" not in st.session_state:
    st.session_state.medium_priority = 0

if "low_priority" not in st.session_state:
    st.session_state.low_priority = 0

if "review_count" not in st.session_state:
    st.session_state.review_count = 0


# ============================================================
# 5. APPLICATION CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 60.0

queue_mapping = {

    "Delivery":
        "Delivery Support",

    "Payment":
        "Billing & Payments",

    "Account":
        "Account Support",

    "Return":
        "Returns Team",

    "Product Issue":
        "Product Support",

    "Order Cancellation":
        "Order Management"

}


# ============================================================
# 6. PAGE HEADER
# ============================================================

st.title(
    "🎫 AI Customer Support Ticket Triage"
)

st.write(
    "AI-powered customer support ticket classification, "
    "priority detection, support routing, and "
    "human-review management."
)

st.divider()


# ============================================================
# 7. DASHBOARD OVERVIEW
# ============================================================

st.subheader(
    "📊 Dashboard Overview"
)

metric1, metric2, metric3, metric4, metric5 = st.columns(5)


with metric1:

    st.metric(
        "🎫 Tickets Analyzed",
        st.session_state.tickets_analyzed
    )


with metric2:

    st.metric(
        "🔴 High Priority",
        st.session_state.high_priority
    )


with metric3:

    st.metric(
        "🟠 Medium Priority",
        st.session_state.medium_priority
    )


with metric4:

    st.metric(
        "🟢 Low Priority",
        st.session_state.low_priority
    )


with metric5:

    # Read actual review queue count
    if REVIEW_LOG_PATH.exists():

        try:

            review_df = pd.read_csv(
                REVIEW_LOG_PATH
            )

            review_count = len(
                review_df
            )

        except Exception:

            review_count = (
                st.session_state.review_count
            )

    else:

        review_count = (
            st.session_state.review_count
        )


    st.metric(
        "👨‍💼 Human Review",
        review_count
    )


st.divider()


# ============================================================
# 8. HUMAN REVIEW LOG FUNCTION
# ============================================================

def log_for_human_review(
    customer,
    ticket,
    category,
    category_confidence,
    priority,
    priority_confidence,
    queue
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # --------------------------------------------------------
    # Check whether review log already exists
    # --------------------------------------------------------

    if REVIEW_LOG_PATH.exists():

        try:

            existing_df = pd.read_csv(
                REVIEW_LOG_PATH
            )

        except Exception:

            existing_df = pd.DataFrame()

    else:

        existing_df = pd.DataFrame()


    # --------------------------------------------------------
    # Prevent duplicate tickets
    # --------------------------------------------------------

    if not existing_df.empty:

        required_log_columns = [
            "customer",
            "ticket_text",
            "status"
        ]

        if all(
            column in existing_df.columns
            for column in required_log_columns
        ):

            duplicate = existing_df[
                (
                    existing_df["customer"]
                    .astype(str)
                    ==
                    str(customer)
                )
                &
                (
                    existing_df["ticket_text"]
                    .astype(str)
                    ==
                    str(ticket)
                )
                &
                (
                    existing_df["status"]
                    .astype(str)
                    ==
                    "Pending Review"
                )
            ]


            if not duplicate.empty:

                return False


    # --------------------------------------------------------
    # Create new review record
    # --------------------------------------------------------

    new_record = {

        "timestamp":
            timestamp,

        "customer":
            customer,

        "ticket_text":
            ticket,

        "predicted_category":
            category,

        "category_confidence":
            round(
                category_confidence,
                2
            ),

        "predicted_priority":
            priority,

        "priority_confidence":
            round(
                priority_confidence,
                2
            ),

        "recommended_queue":
            queue,

        "status":
            "Pending Review"

    }


    new_df = pd.DataFrame(
        [new_record]
    )


    # --------------------------------------------------------
    # Append new record
    # --------------------------------------------------------

    if not existing_df.empty:

        updated_df = pd.concat(
            [
                existing_df,
                new_df
            ],
            ignore_index=True
        )

    else:

        updated_df = new_df


    updated_df.to_csv(
        REVIEW_LOG_PATH,
        index=False
    )


    return True


# ============================================================
# 9. SUBMIT CUSTOMER TICKET
# ============================================================

st.subheader(
    "📩 Submit a Customer Support Ticket"
)


customer_name = st.text_input(
    "Customer Name",
    placeholder="Enter customer name"
)


ticket_text = st.text_area(
    "Describe your issue",

    placeholder=(
        "Example: My card was charged twice "
        "for the same order. Please help immediately."
    ),

    height=150
)


# ============================================================
# 10. ANALYZE TICKET
# ============================================================

if st.button(
    "🔍 Analyze Ticket",
    type="primary"
):

    # --------------------------------------------------------
    # Validate ticket
    # --------------------------------------------------------

    if not ticket_text.strip():

        st.warning(
            "⚠️ Please enter a ticket description."
        )


    else:

        ticket_input = [
            ticket_text.strip()
        ]


        # ====================================================
        # 10.1 CATEGORY PREDICTION
        # ====================================================

        predicted_category = (
            category_model
            .predict(ticket_input)[0]
        )


        # ====================================================
        # 10.2 PRIORITY PREDICTION
        # ====================================================

        predicted_priority = (
            priority_model
            .predict(ticket_input)[0]
        )


        # ====================================================
        # 10.3 CATEGORY PROBABILITIES
        # ====================================================

        category_probabilities = (
            category_model
            .predict_proba(ticket_input)[0]
        )


        # ====================================================
        # 10.4 PRIORITY PROBABILITIES
        # ====================================================

        priority_probabilities = (
            priority_model
            .predict_proba(ticket_input)[0]
        )


        # ====================================================
        # 10.5 CONFIDENCE SCORES
        # ====================================================

        category_confidence = (
            max(category_probabilities)
            * 100
        )


        priority_confidence = (
            max(priority_probabilities)
            * 100
        )


        lowest_confidence = min(
            category_confidence,
            priority_confidence
        )


        # ====================================================
        # 10.6 SUPPORT QUEUE
        # ====================================================

        recommended_queue = queue_mapping.get(
            predicted_category,
            "General Support"
        )


        # ====================================================
        # 10.7 UPDATE SESSION STATISTICS
        # ====================================================

        st.session_state.tickets_analyzed += 1


        if predicted_priority == "High":

            st.session_state.high_priority += 1

        elif predicted_priority == "Medium":

            st.session_state.medium_priority += 1

        elif predicted_priority == "Low":

            st.session_state.low_priority += 1


        # ====================================================
        # 11. SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ Ticket analyzed successfully!"
        )

        st.divider()


        # ====================================================
        # 12. TICKET DETAILS
        # ====================================================

        st.subheader(
            "📋 Ticket Details"
        )


        detail_col1, detail_col2 = st.columns(2)


        with detail_col1:

            st.write(
                "**Customer:**",
                customer_name or "Anonymous"
            )


        with detail_col2:

            st.write(
                "**Ticket Length:**",
                f"{len(ticket_text.strip())} characters"
            )


        st.write(
            "**Ticket Description:**"
        )


        st.info(
            ticket_text.strip()
        )


        st.divider()


        # ====================================================
        # 13. AI CLASSIFICATION
        # ====================================================

        st.subheader(
            "🤖 AI Classification"
        )


        classification_col1, classification_col2 = (
            st.columns(2)
        )


        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        with classification_col1:

            st.metric(
                "📂 Predicted Category",
                predicted_category
            )

            st.write(
                f"**Category Confidence: "
                f"{category_confidence:.2f}%**"
            )

            st.progress(
                min(
                    int(category_confidence),
                    100
                )
            )


        # ----------------------------------------------------
        # PRIORITY
        # ----------------------------------------------------

        with classification_col2:

            st.metric(
                "🚨 Predicted Priority",
                predicted_priority
            )

            st.write(
                f"**Priority Confidence: "
                f"{priority_confidence:.2f}%**"
            )

            st.progress(
                min(
                    int(priority_confidence),
                    100
                )
            )


        st.divider()


        # ====================================================
        # 14. SUPPORT ROUTING
        # ====================================================

        st.subheader(
            "📌 Recommended Support Queue"
        )


        st.info(
            f"🎯 **{recommended_queue}**"
        )


        st.divider()


        # ====================================================
        # 15. CONFIDENCE ANALYSIS
        # ====================================================

        st.subheader(
            "🔎 Prediction Confidence Analysis"
        )


        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        if lowest_confidence < CONFIDENCE_THRESHOLD:

            st.error(
                f"🚨 **Human Review Required**\n\n"
                f"Lowest prediction confidence: "
                f"{lowest_confidence:.2f}%\n\n"
                f"The ticket is below the "
                f"{CONFIDENCE_THRESHOLD:.0f}% "
                f"confidence threshold."
            )


            # ------------------------------------------------
            # Save to human review queue
            # ------------------------------------------------

            was_added = log_for_human_review(

                customer_name or "Anonymous",

                ticket_text.strip(),

                predicted_category,

                category_confidence,

                predicted_priority,

                priority_confidence,

                recommended_queue

            )


            if was_added:

                st.session_state.review_count += 1

                st.warning(
                    "📥 Ticket added to the "
                    "Human Review Queue."
                )

            else:

                st.info(
                    "ℹ️ This ticket is already "
                    "present in the Human Review Queue."
                )


        # ----------------------------------------------------
        # MODERATE CONFIDENCE
        # ----------------------------------------------------

        elif lowest_confidence < 80:

            st.warning(
                f"⚠️ **Moderate Confidence**\n\n"
                f"Lowest prediction confidence: "
                f"{lowest_confidence:.2f}%\n\n"
                "The ticket can be processed, "
                "but manual verification is recommended."
            )


        # ----------------------------------------------------
        # HIGH CONFIDENCE
        # ----------------------------------------------------

        else:

            st.success(
                f"✅ **High-Confidence Prediction**\n\n"
                f"Lowest prediction confidence: "
                f"{lowest_confidence:.2f}%\n\n"
                "The ticket can be processed automatically."
            )


        st.divider()


# ============================================================
# 16. HUMAN REVIEW QUEUE
# ============================================================

st.subheader(
    "👨‍💼 Human Review Queue"
)


if REVIEW_LOG_PATH.exists():

    try:

        review_df = pd.read_csv(
            REVIEW_LOG_PATH
        )


        if not review_df.empty:

            st.write(
                f"Pending review tickets: "
                f"**{len(review_df)}**"
            )


            # Show latest tickets first
            display_df = (
                review_df
                .tail(10)
                .iloc[::-1]
            )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


        else:

            st.info(
                "No tickets are currently waiting "
                "for human review."
            )


    except Exception as error:

        st.warning(
            f"Unable to read review queue: "
            f"{error}"
        )


else:

    st.info(
        "No tickets have been flagged for "
        "human review yet."
    )


# ============================================================
# 17. SYSTEM INFORMATION
# ============================================================

st.divider()


st.subheader(
    "⚙️ System Information"
)


info_col1, info_col2, info_col3 = (
    st.columns(3)
)


with info_col1:

    st.write(
        "**Text Processing:** TF-IDF"
    )


with info_col2:

    st.write(
        "**Classification:** Logistic Regression"
    )


with info_col3:

    st.write(
        f"**Review Threshold:** "
        f"{CONFIDENCE_THRESHOLD:.0f}%"
    )


st.caption(
    "AI Customer Support Ticket Triage | "
    "Machine Learning + Streamlit"
)