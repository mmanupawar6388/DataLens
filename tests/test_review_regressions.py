import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from modules.assistant import answer_question
from modules.data_context import build_dataset_context
from modules.eda import sales_by_category, sales_by_city


class ReviewRegressionTests(unittest.TestCase):
    def test_empty_dataset_after_drop_shows_message_before_sales_metrics(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()
        app.file_uploader[0].upload(
            "sales.csv",
            b"order_id,quantity,unit_price,revenue,city\n,1,1,1,Mysuru\n",
            "text/csv",
        ).run()
        app.selectbox[0].set_value("drop")
        next(
            button for button in app.button
            if "Handle Missing Values" in button.label
        ).click()
        app.run()

        self.assertTrue(
            any("No rows remain after data cleaning." in item.value for item in app.warning)
        )
        self.assertFalse(app.exception)
        self.assertFalse(any(metric.label == "Total Revenue" for metric in app.metric))

    def test_unsupported_non_revenue_filter_does_not_return_unfiltered_total(self):
        df = pd.DataFrame({
            "city": ["Mysuru", "Bengaluru"],
            "quantity": [2, 8],
            "revenue": [20, 80],
        })

        response = answer_question(df, "What is the total quantity for Mysuru?")

        self.assertIn("Filters are not supported", response["text"])
        self.assertNotIn("10", response["text"])

    def test_unfiltered_count_in_total_keeps_existing_overview_behavior(self):
        df = pd.DataFrame({"quantity": [2, 8]})

        response = answer_question(df, "How many rows in total?")

        self.assertEqual(response["table_title"], "Dataset Statistics")

    def test_gemini_context_omits_raw_categorical_values_and_small_numeric_values(self):
        df = pd.DataFrame({
            "email": ["private@example.test"],
            "score": [987654],
        })

        context = build_dataset_context(df)

        self.assertNotIn("private@example.test", context)
        self.assertNotIn("987654", context)
        self.assertIn("Numeric aggregates omitted", context)

    def test_dashboard_and_assistant_include_missing_category_and_city_revenue(self):
        df = pd.DataFrame({
            "category": ["Stationery", None],
            "city": ["Mysuru", None],
            "revenue": [20, 80],
        })

        dashboard_category = sales_by_category(df)
        dashboard_city = sales_by_city(df)
        assistant_category = (
            answer_question(df, "revenue by category")["table"]
            .set_index("Category")["Total Revenue"]
        )
        assistant_city = (
            answer_question(df, "revenue by city")["table"]
            .set_index("City")["Total Revenue"]
        )

        self.assertEqual(dashboard_category.to_dict(), assistant_category.to_dict())
        self.assertEqual(dashboard_city.to_dict(), assistant_city.to_dict())
        self.assertEqual(dashboard_category.sum(), 100)
        self.assertEqual(dashboard_city.sum(), 100)

    def test_combined_category_ranking_and_breakdown_share_one_table(self):
        df = pd.DataFrame({
            "category": ["Stationery", "Home"],
            "revenue": [20, 10],
        })

        response = answer_question(
            df,
            "Which product category generates the highest revenue, "
            "and what is the total revenue contribution of each category?",
        )

        self.assertIn("Stationery", response["text"])
        self.assertIn("20.00", response["text"])
        self.assertEqual(response["table_title"], "Revenue by Category")
        self.assertEqual(response["table"]["Total Revenue"].tolist(), [20, 10])

    def test_combined_breakdowns_request_separate_tables(self):
        df = pd.DataFrame({
            "category": ["Stationery"],
            "city": ["Mysuru"],
            "revenue": [20],
        })

        response = answer_question(df, "revenue by city and revenue by category")

        self.assertIsNone(response["table"])
        self.assertIn("separately", response["text"])


if __name__ == "__main__":
    unittest.main()
