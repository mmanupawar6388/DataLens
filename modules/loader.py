import pandas as pd


def load_csv(file):
    try:
        df = pd.read_csv(file)

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        return df

    except Exception as e:
        raise ValueError(f"Could not load CSV file: {e}")