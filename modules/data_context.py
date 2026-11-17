
import pandas as pd

MIN_AGGREGATE_VALUES = 5


def build_dataset_context(df):
    """Build aggregate-only dataset context for Gemini."""

    if df is None or df.empty:
        return "No dataset is currently available."

    context = []

    context.append(f"Record rows: {len(df)}")
    context.append(f"Total columns: {len(df.columns)}")
    context.append(
        f"Exact duplicate rows: {int(df.duplicated().sum())}; "
        "duplicate rows are reported, not automatically removed."
    )

    order_id_column = next(
        (
            col for col in df.columns
            if str(col).strip().casefold().replace(" ", "_") == "order_id"
        ),
        None
    )
    if order_id_column is not None:
        unique_orders = int(df[order_id_column].nunique(dropna=True))
        context.append(
            f"Unique orders with a non-missing {order_id_column}: {unique_orders}"
        )
    else:
        context.append(
            "Unique order count is unavailable because there is no order_id column; "
            "do not describe record rows as orders."
        )

    context.append("\nCOLUMN INFORMATION:")

    for col in df.columns:
        series = df[col]

        context.append(
            f"- {col}: {series.dtype}, "
            f"{series.nunique()} unique values, "
            f"{series.isnull().sum()} missing values"
        )

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        context.append(
            "\nNUMERICAL SUMMARY (only columns with at least "
            f"{MIN_AGGREGATE_VALUES} non-missing values):"
        )
        summarized_columns = 0

        for col in numeric_df.columns:
            values = numeric_df[col].dropna()

            if len(values) < MIN_AGGREGATE_VALUES:
                continue

            summarized_columns += 1
            context.append(
                f"\nColumn: {col}"
                f"\nAverage: {values.mean():.2f}"
                f"\nMedian: {values.median():.2f}"
                f"\nMinimum: {values.min():.2f}"
                f"\nMaximum: {values.max():.2f}"
                f"\nTotal: {values.sum():.2f}"
            )

        if not summarized_columns:
            context.append("Numeric aggregates omitted: too few values.")

    return "\n".join(context)
