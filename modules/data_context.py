
import pandas as pd


def build_dataset_context(df):
    """Build a privacy-conscious summary for Gemini."""

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
        context.append("\nNUMERICAL SUMMARY:")

        for col in numeric_df.columns:
            values = numeric_df[col].dropna()

            if values.empty:
                continue

            context.append(
                f"\nColumn: {col}"
                f"\nAverage: {values.mean():.2f}"
                f"\nMedian: {values.median():.2f}"
                f"\nMinimum: {values.min():.2f}"
                f"\nMaximum: {values.max():.2f}"
                f"\nTotal: {values.sum():.2f}"
            )

    context.append("\nCATEGORICAL SUMMARY:")

    for col in df.select_dtypes(exclude="number").columns:

        # Avoid sending values from high-cardinality columns,
        # which may contain names, IDs, or other personal data.
        if df[col].nunique() <= 20:
            all_counts = df[col].value_counts()
            counts = all_counts.head(5)
            if not counts.empty:
                counts = all_counts[all_counts >= counts.iloc[-1]]

            context.append(
                f"\nColumn: {col}\nTop values (up to 5; ties at the cutoff included):\n"
                f"{counts.to_string()}"
            )

    return "\n".join(context)
