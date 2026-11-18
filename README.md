# DataLens

**CSV Analysis, Data Cleaning, Exploratory Data Analysis and AI Assistant**

DataLens is a Streamlit-based application that helps users upload CSV datasets, understand their structure, clean data, explore trends and ask questions using a Gemini-powered AI assistant.

**Live demo:** [Try DataLens](https://datalens-b8ptmnzbhk2s8f2nii32d8.streamlit.app/)

## Features

- CSV file upload and preview
- Automated dataset profiling
- Missing-value and duplicate detection
- Data cleaning and cleaned CSV export
- Exploratory data analysis and visualizations
- AI-powered natural-language questions about datasets
- Dataset-based business insights

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Google Gemini API
- Matplotlib / Plotly (where used)

## Installation

1. Clone the repository.
2. Install the dependencies:

   `python -m pip install -r requirements.txt`

3. Create a `.env` file in the project root:

   `GEMINI_API_KEY=your_api_key_here`

4. Start the application:

   `streamlit run app.py`

## Security

Never commit your `.env` file or expose your Gemini API key.

For unsupported questions, DataLens sends your question and a dataset summary to Gemini. The summary contains column names and types, row, duplicate, missing, and unique-value counts, and numeric aggregates only for columns with at least five non-missing values. Raw rows and categorical values are not sent.

Do not upload confidential or sensitive datasets to the AI assistant without an appropriate data privacy policy.

## Project Status

DataLens v1.0 — Development and testing in progress.

## Author

Manu Pawar
