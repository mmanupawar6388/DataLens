# ============================================================
# app.py
# DataLens - Dataset Profiling, Cleaning & EDA
# ============================================================

import hashlib

import streamlit as st

from modules.assistant import render_assistant

from modules.loader import load_csv

from modules.profiler import (
    get_dataset_profile,
    get_column_profile
)

from modules.cleaner import (
    remove_duplicates,
    fill_missing_values
)

from modules.eda import (
    get_kpis,
    get_business_insights,
    sales_by_category,
    sales_by_city,
    price_distribution,
    sales_over_time,
    quantity_distribution
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DataLens",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "dataset_hash" not in st.session_state:
    st.session_state.dataset_hash = None


def clear_assistant_history():
    st.session_state.assistant_messages = []


# ============================================================
# HEADER
# ============================================================

st.title("📊 DataLens")

st.write(
    "Upload your dataset and get an instant overview."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.session_state.df = None
    st.session_state.dataset_hash = None
    clear_assistant_history()

    st.info(
        "Upload a CSV file to begin."
    )

    st.stop()


# ============================================================
# LOAD NEW DATASET
# ============================================================

current_dataset_hash = hashlib.sha256(
    uploaded_file.getvalue()
).hexdigest()

if st.session_state.dataset_hash != current_dataset_hash:

    try:

        loaded_df = load_csv(
            uploaded_file
        )

        st.session_state.df = loaded_df
        st.session_state.dataset_hash = current_dataset_hash
        clear_assistant_history()

    except ValueError as e:

        st.error(str(e))
        st.stop()

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )
        st.stop()


# ============================================================
# CURRENT DATAFRAME
# ============================================================

df = st.session_state.df

st.success(
    "Dataset loaded successfully!"
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

profile = get_dataset_profile(df)

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        profile["rows"]
    )

with col2:
    st.metric(
        "Columns",
        profile["columns"]
    )

with col3:
    st.metric(
        "Missing Values",
        profile["missing_values"]
    )

with col4:
    st.metric(
        "Duplicate Rows",
        profile["duplicate_rows"]
    )


# ============================================================
# KEY METRICS
# ============================================================

st.header("📌 Key Metrics")

sales_columns = {"quantity", "unit_price", "revenue"}

if sales_columns.issubset(df.columns):

    kpis = get_kpis(df)

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Total Revenue",
            f"₹{kpis['total_revenue']:,.0f}"
        )

    with metric2:
        if kpis["total_orders"] is None:
            st.metric(
                "Rows",
                f"{kpis['row_count']:,}"
            )
        else:
            st.metric(
                "Unique Orders",
                f"{kpis['total_orders']:,}"
            )

    with metric3:
        st.metric(
            "Units Sold",
            f"{kpis['units_sold']:,.0f}"
        )

    with metric4:
        average_value = (
            kpis["average_order_value"]
            if kpis["total_orders"] is not None
            else kpis["average_revenue_per_row"]
        )
        average_label = (
            "Average Order Value"
            if kpis["total_orders"] is not None
            else "Average Revenue per Row"
        )
        average_help = (
            "Excludes exact duplicate rows to align revenue with unique orders. "
            "Total Revenue still includes all uploaded rows."
            if kpis["total_orders"] is not None
            else None
        )
        st.metric(
            average_label,
            f"₹{average_value:,.0f}",
            help=average_help
        )

else:
    st.info(
        "Sales metrics are unavailable. "
        "Upload a sales dataset containing revenue, "
        "quantity, or unit_price columns to view them."
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.header("Data Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# COLUMN INFORMATION
# ============================================================

st.header("Column Information")

column_profile = get_column_profile(df)

st.dataframe(
    column_profile,
    use_container_width=True
)


# ============================================================
# DATA CLEANING
# ============================================================

st.header("🧹 Data Cleaning")


# ------------------------------------------------------------
# DUPLICATE ROWS
# ------------------------------------------------------------

if profile["duplicate_rows"] > 0:

    st.warning(
        f"This dataset contains "
        f"{profile['duplicate_rows']} duplicate row(s)."
    )

    if st.button("🧹 Remove Duplicate Rows"):

        cleaned_df, removed = remove_duplicates(df)

        if not df.equals(cleaned_df):
            clear_assistant_history()

        st.session_state.df = cleaned_df

        st.success(
            f"{removed} duplicate row(s) removed successfully!"
        )

        st.rerun()

else:

    st.success(
        "No duplicate rows found."
    )


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

if profile["missing_values"] > 0:

    st.warning(
        f"This dataset contains "
        f"{profile['missing_values']} missing value(s)."
    )

    method = st.selectbox(
        "How would you like to handle missing values?",
        [
            "median",
            "mean",
            "drop"
        ]
    )

    if st.button("🧹 Handle Missing Values"):

        cleaned_df = fill_missing_values(
            df,
            method
        )

        if not df.equals(cleaned_df):
            clear_assistant_history()

        st.session_state.df = cleaned_df

        st.success(
            f"Missing values handled using: {method}"
        )

        st.rerun()

else:

    st.success(
        "No missing values found."
    )


# ============================================================
# EXPORT CLEANED DATASET
# ============================================================

st.header("📥 Export Cleaned Dataset")

st.write(
    "Download the current dataset after applying "
    "your cleaning operations."
)

cleaned_df = st.session_state.df

cleaned_csv = cleaned_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Cleaned CSV",
    data=cleaned_csv,
    file_name="datalens_cleaned_dataset.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

st.header("📊 Exploratory Data Analysis")


# ============================================================
# AUTOMATIC INSIGHTS
# ============================================================

insights = get_business_insights(df)

if insights:

    st.subheader("💡 Automatic Insights")

    for insight in insights:

        st.info(
            f"• {insight}"
        )


# ============================================================
# REVENUE BY CATEGORY
# ============================================================

category_data = sales_by_category(df)

if not category_data.empty:

    st.subheader("Revenue by Category")

    st.bar_chart(
        category_data,
        horizontal=True,
        use_container_width=True
    )

else:

    st.info(
        "Category analysis requires a 'category' column."
    )


# ============================================================
# REVENUE BY CITY
# ============================================================

city_data = sales_by_city(df)

if not city_data.empty:

    st.subheader("Revenue by City")

    st.bar_chart(
        city_data,
        horizontal=True,
        use_container_width=True
    )

else:

    st.info(
        "City analysis requires a 'city' column."
    )


# ============================================================
# PRICE DISTRIBUTION
# ============================================================

price_data = price_distribution(df)

if not price_data.empty:

    st.subheader("Price Distribution")

    st.bar_chart(
        price_data,
        use_container_width=True
    )

else:

    st.info(
        "Price Distribution requires valid "
        "'unit_price' data."
    )


# ============================================================
# REVENUE OVER TIME
# ============================================================

time_data = sales_over_time(df)

if not time_data.empty:

    st.subheader("Revenue Over Time")

    st.line_chart(
        time_data,
        use_container_width=True
    )

else:

    st.info(
        "Revenue Over Time requires valid "
        "'purchase_date', 'quantity', and "
        "'unit_price' data."
    )


# ============================================================
# QUANTITY DISTRIBUTION
# ============================================================

quantity_data = quantity_distribution(df)

if not quantity_data.empty:

    st.subheader("Quantity Distribution")

    st.bar_chart(
        quantity_data,
        use_container_width=True
    )

else:

    st.info(
        "Quantity Distribution requires valid "
        "'quantity' data."
    )

# ============================================================
# DATALENS AI ASSISTANT
# ============================================================

st.divider()

st.header("🤖 DataLens AI Assistant")

render_assistant(df)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DataLens • Automated Dataset Profiling, Cleaning & EDA"
)
