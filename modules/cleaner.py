import pandas as pd


def remove_duplicates(df):
    cleaned_df = df.drop_duplicates().reset_index(drop=True)

    removed = len(df) - len(cleaned_df)

    return cleaned_df, removed


def fill_missing_values(df, method):
    cleaned_df = df.copy()

    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns
    categorical_columns = cleaned_df.select_dtypes(
        exclude="number"
    ).columns

    if method in ["mean", "median"]:
        for column in numeric_columns:
            if cleaned_df[column].isnull().any():
                if method == "mean":
                    fill_value = cleaned_df[column].mean()
                else:
                    fill_value = cleaned_df[column].median()
                cleaned_df[column] = cleaned_df[column].fillna(
                    fill_value
                )

        for column in categorical_columns:
            cleaned_df[column] = cleaned_df[column].fillna("Unknown")

    elif method == "drop":
        cleaned_df = cleaned_df.dropna().reset_index(drop=True)

    return cleaned_df