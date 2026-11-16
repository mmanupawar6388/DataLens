def get_dataset_profile(df):
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

    return profile


def get_column_profile(df):
    columns = []

    for column in df.columns:
        columns.append({
            "column": column,
            "type": str(df[column].dtype),
            "missing": int(df[column].isnull().sum()),
            "unique": int(df[column].nunique())
        })

    return columns