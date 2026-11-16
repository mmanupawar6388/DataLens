# =========================================================
# eda.py
# DataLens - Exploratory Data Analysis
# =========================================================

import pandas as pd


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _prepare_data(df):
    """
    Prepare a safe copy of the dataset for EDA calculations.

    Does not modify the original dataframe.
    """

    if df is None:
        return pd.DataFrame()

    data = df.copy()

    # Convert numeric columns
    for col in ["quantity", "unit_price", "revenue"]:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            ).replace([float("inf"), -float("inf")], float("nan"))

    # Convert date column
    if "purchase_date" in data.columns:
        data["purchase_date"] = pd.to_datetime(
            data["purchase_date"],
            errors="coerce"
        )

    # Preserve supplied revenue (including discounts); derive only if absent.
    if (
        "revenue" not in data.columns
        and "quantity" in data.columns
        and "unit_price" in data.columns
    ):
        data["revenue"] = (
            data["quantity"] * data["unit_price"]
        )

    if "revenue" in data.columns:
        data["revenue"] = data["revenue"].replace(
            [float("inf"), -float("inf")], float("nan")
        )

    return data


def _find_order_id_column(data):
    return next(
        (
            col for col in data.columns
            if str(col).strip().casefold().replace(" ", "_") == "order_id"
        ),
        None
    )


# =========================================================
# KPI CALCULATIONS
# =========================================================

def get_kpis(df):
    """
    Return the main business KPIs.
    """

    data = _prepare_data(df)

    total_revenue = 0
    row_count = len(data)
    order_id_column = _find_order_id_column(data)
    has_complete_order_ids = (
        order_id_column is not None
        and not data[order_id_column].isna().any()
    )
    total_orders = (
        int(data[order_id_column].nunique(dropna=True))
        if has_complete_order_ids
        else None
    )
    units_sold = 0
    average_order_value = None
    average_revenue_per_row = 0

    if "revenue" in data.columns:
        total_revenue = data["revenue"].sum(min_count=1)

    if "quantity" in data.columns:
        units_sold = data["quantity"].sum(min_count=1)

    if "revenue" in data.columns and data["revenue"].isna().any():
        average_revenue_per_row = float("nan")
    elif row_count > 0:
        average_revenue_per_row = total_revenue / row_count

    if total_orders is not None and "revenue" in data.columns:
        order_data = data.drop_duplicates()
        order_revenue = order_data["revenue"].sum(min_count=1)
        if order_data["revenue"].isna().any():
            average_order_value = float("nan")
        elif total_orders > 0:
            average_order_value = order_revenue / total_orders

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "row_count": row_count,
        "units_sold": units_sold,
        "average_order_value": average_order_value,
        "average_revenue_per_row": average_revenue_per_row
    }


# =========================================================
# AUTOMATIC BUSINESS INSIGHTS
# =========================================================

def get_business_insights(df):
    """
    Generate simple automatic business insights.
    """

    data = _prepare_data(df)

    insights = []

    if "revenue" in data.columns:
        missing_revenue = int(data["revenue"].isna().sum())
        if missing_revenue:
            order_id_column = _find_order_id_column(data)
            average_label = (
                "Average order value"
                if (
                    order_id_column is not None
                    and not data[order_id_column].isna().any()
                )
                else "Average revenue per row"
            )
            insights.append(
                f"Revenue is missing or invalid in {missing_revenue} row(s). "
                f"Revenue totals exclude these rows and may be incomplete. "
                f"{average_label} is unavailable until they are resolved."
            )

    # Highest revenue category
    if (
        "category" in data.columns
        and "revenue" in data.columns
    ):
        category_revenue = (
            data.groupby("category")["revenue"]
            .sum(min_count=1)
            .dropna()
            .sort_values(ascending=False)
        )

        if not category_revenue.empty:
            top_revenue = category_revenue.max()
            top_categories = category_revenue[
                category_revenue.eq(top_revenue)
            ].index
            category_names = ", ".join(map(str, top_categories))

            insights.append(
                f"{category_names} "
                f"{'tied for the highest' if len(top_categories) > 1 else 'generated the highest'} "
                f"revenue at ₹{top_revenue:,.0f}."
            )

    # Highest revenue city
    if (
        "city" in data.columns
        and "revenue" in data.columns
    ):
        city_revenue = (
            data.groupby("city")["revenue"]
            .sum(min_count=1)
            .dropna()
            .sort_values(ascending=False)
        )

        if not city_revenue.empty:
            top_city_revenue = city_revenue.max()
            top_cities = city_revenue[
                city_revenue.eq(top_city_revenue)
            ].index
            city_names = ", ".join(map(str, top_cities))

            insights.append(
                f"{city_names} "
                f"{'tied for the highest' if len(top_cities) > 1 else 'generated the highest'} "
                f"city revenue at ₹{top_city_revenue:,.0f}."
            )

    # Most common quantity
    if "quantity" in data.columns:
        quantity_data = data["quantity"].dropna()

        if not quantity_data.empty:
            mode_values = quantity_data.mode()

            if not mode_values.empty:
                common_quantity = mode_values.iloc[0]

                if float(common_quantity).is_integer():
                    common_quantity = int(common_quantity)

                insights.append(
                    f"The most common recorded quantity is "
                    f"{common_quantity} unit(s)."
                )

    # Highest row or order value
    if "revenue" in data.columns:
        order_id_column = _find_order_id_column(data)
        if (
            order_id_column is not None
            and not data[order_id_column].isna().any()
        ):
            order_values = (
                data.groupby(order_id_column)["revenue"]
                .sum(min_count=1)
                .dropna()
            )
            value_label = "order"
        else:
            order_values = data["revenue"].dropna()
            value_label = "row"

        if not order_values.empty:
            highest_order = order_values.max()

            insights.append(
                f"The highest recorded {value_label} revenue is "
                f"₹{highest_order:,.0f}."
            )

    return insights


# =========================================================
# REVENUE BY CATEGORY
# =========================================================

def sales_by_category(df):
    """
    Calculate total revenue for each category.

    Returns:
        pandas Series
    """

    data = _prepare_data(df)

    if (
        "category" not in data.columns
        or "revenue" not in data.columns
    ):
        return pd.Series(dtype="float64")

    return (
        data.groupby("category")["revenue"]
        .sum(min_count=1)
        .sort_values(ascending=False)
    )


# =========================================================
# REVENUE BY CITY
# =========================================================

def sales_by_city(df):
    """
    Calculate total revenue for each city.

    Returns:
        pandas Series
    """

    data = _prepare_data(df)

    if (
        "city" not in data.columns
        or "revenue" not in data.columns
    ):
        return pd.Series(dtype="float64")

    return (
        data.groupby("city")["revenue"]
        .sum(min_count=1)
        .sort_values(ascending=False)
    )


# =========================================================
# PRICE DISTRIBUTION
# =========================================================

def price_distribution(df):
    """
    Return the number of record rows in each price range.

    Returns:
        pandas Series
    """

    if (
        df is None
        or "unit_price" not in df.columns
    ):
        return pd.Series(dtype="int64")

    prices = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    ).dropna()

    # Remove invalid negative prices
    prices = prices[prices >= 0]

    if prices.empty:
        return pd.Series(dtype="int64")

    # Fixed business-friendly ranges
    bins = [
        0,
        1000,
        2500,
        5000,
        7500,
        float("inf")
    ]

    labels = [
        "₹0 – ₹1,000",
        "₹1,000 – ₹2,500",
        "₹2,500 – ₹5,000",
        "₹5,000 – ₹7,500",
        "₹7,500+"
    ]

    price_ranges = pd.cut(
        prices,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    return (
        price_ranges
        .value_counts(sort=False)
        .reindex(labels, fill_value=0)
    )


# =========================================================
# REVENUE OVER TIME
# =========================================================

def sales_over_time(df):
    """
    Calculate total revenue for each purchase date.

    Returns:
        pandas Series
    """

    if df is None:
        return pd.Series(dtype="float64")

    data = _prepare_data(df)

    if not {"purchase_date", "revenue"}.issubset(data.columns):
        return pd.Series(dtype="float64")

    # Remove invalid rows
    data = data.dropna(
        subset=[
            "purchase_date",
            "revenue"
        ]
    )

    if data.empty:
        return pd.Series(dtype="float64")

    return (
        data.groupby("purchase_date")["revenue"]
        .sum(min_count=1)
        .sort_index()
    )


# =========================================================
# QUANTITY DISTRIBUTION
# =========================================================

def quantity_distribution(df):
    """
    Calculate the number of record rows for each quantity value.

    Returns:
        pandas Series
    """

    if (
        df is None
        or "quantity" not in df.columns
    ):
        return pd.Series(dtype="int64")

    quantity = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    ).dropna()

    if quantity.empty:
        return pd.Series(dtype="int64")

    return (
        quantity
        .value_counts()
        .sort_index()
    )


# =========================================================
# CATEGORY SUMMARY
# =========================================================

def category_summary(df):
    """
    Return a category-level summary containing:

    - Unique Orders when order IDs are available, otherwise Rows
    - Units Sold
    - Revenue
    - Average Price
    """

    data = _prepare_data(df)

    required = [
        "category",
        "quantity",
        "unit_price",
        "revenue"
    ]

    if not all(
        col in data.columns
        for col in required
    ):
        return pd.DataFrame()

    order_id_column = _find_order_id_column(data)
    order_count = (
        {"Orders": (order_id_column, "nunique")}
        if order_id_column is not None and not data[order_id_column].isna().any()
        else {"Rows": ("category", "size")}
    )

    return (
        data.groupby("category")
        .agg(
            **order_count,
            Units_Sold=("quantity", "sum"),
            Revenue=("revenue", lambda values: values.sum(min_count=1)),
            Average_Price=("unit_price", "mean")
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
    )


# =========================================================
# CITY SUMMARY
# =========================================================

def city_summary(df):
    """
    Return a city-level summary containing:

    - Unique Orders when order IDs are available, otherwise Rows
    - Units Sold
    - Revenue
    - Average Order Value
    """

    data = _prepare_data(df)

    required = [
        "city",
        "quantity",
        "revenue"
    ]

    if not all(
        col in data.columns
        for col in required
    ):
        return pd.DataFrame()

    order_id_column = _find_order_id_column(data)
    order_count = (
        {"Orders": (order_id_column, "nunique")}
        if order_id_column is not None and not data[order_id_column].isna().any()
        else {"Rows": ("city", "size")}
    )
    count_column = "Orders" if "Orders" in order_count else "Rows"
    average_column = (
        "Average_Order_Value"
        if count_column == "Orders"
        else "Average_Revenue_Per_Row"
    )

    result = (
        data.groupby("city")
        .agg(
            **order_count,
            Units_Sold=("quantity", "sum"),
            Revenue=("revenue", "sum")
        )
    )

    result[average_column] = (
        result["Revenue"]
        / result[count_column]
    )

    return result.sort_values(
        "Revenue",
        ascending=False
    )