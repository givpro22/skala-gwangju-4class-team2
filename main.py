"""
[DAY 2] 종합 실습
"""

import logging

from src.data_loader import (
    clean_dataframes,
    create_visualizations,
    load_dataframes_from_url,
    run_basic_eda,
    run_statistical_analysis,
    train_evaluate_save_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

pandas_df, polars_df = load_dataframes_from_url(
    url,
    pandas_options={
        "header": None,
        "names": cols,
        "skipinitialspace": True,
    },
    polars_options={
        "has_header": False,
        "new_columns": cols,
    },
)

if pandas_df is not None and polars_df is not None:
    pandas_df, polars_df = clean_dataframes(pandas_df, polars_df)

    print("Pandas DataFrame")
    print(pandas_df.head())

    print("\nPolars DataFrame")
    print(polars_df.head())

    run_basic_eda(pandas_df, polars_df)
    run_statistical_analysis(pandas_df)
    train_evaluate_save_model(pandas_df)
    create_visualizations(pandas_df)
else:
    print("데이터를 불러오지 못했습니다.")
