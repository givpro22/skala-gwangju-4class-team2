from typing import Any

import pandas as pd
import polars as pl


def run_pandas_eda(dataframe: pd.DataFrame, top_n: int = 5) -> None:
    """Pandas DataFrame의 기본 탐색 결과를 출력합니다."""
    print("\n========== Pandas EDA ==========")
    print(f"행/열 개수: {dataframe.shape}")

    print("\n[컬럼 타입]")
    print(dataframe.dtypes)

    print("\n[기술 통계]")
    print(dataframe.describe(include="all").transpose())

    print("\n[컬럼별 결측치]")
    print(dataframe.isna().sum())

    print(f"\n중복 행 개수: {dataframe.duplicated().sum()}")

    categorical_columns = dataframe.select_dtypes(
        include=["object", "string", "category"],
    ).columns
    _print_pandas_category_counts(dataframe, categorical_columns, top_n)


def _print_pandas_category_counts(
    dataframe: pd.DataFrame,
    columns: Any,
    top_n: int,
) -> None:
    print(f"\n[범주형 컬럼 상위 {top_n}개 빈도]")
    if len(columns) == 0:
        print("범주형 컬럼이 없습니다.")
        return

    for column in columns:
        print(f"\n- {column}")
        print(dataframe[column].value_counts(dropna=False).head(top_n))


def run_polars_eda(dataframe: pl.DataFrame, top_n: int = 5) -> None:
    """Polars DataFrame의 기본 탐색 결과를 출력합니다."""
    print("\n========== Polars EDA ==========")
    print(f"행/열 개수: {dataframe.shape}")

    print("\n[컬럼 타입]")
    for column, dtype in dataframe.schema.items():
        print(f"{column}: {dtype}")

    print("\n[기술 통계]")
    print(dataframe.describe())

    print("\n[컬럼별 결측치]")
    print(dataframe.null_count())

    print(f"\n중복 행 개수: {dataframe.is_duplicated().sum()}")

    categorical_columns = [
        column
        for column, dtype in dataframe.schema.items()
        if dtype in (pl.String, pl.Categorical)
    ]
    _print_polars_category_counts(dataframe, categorical_columns, top_n)


def _print_polars_category_counts(
    dataframe: pl.DataFrame,
    columns: Any,
    top_n: int,
) -> None:
    print(f"\n[범주형 컬럼 상위 {top_n}개 빈도]")
    if not columns:
        print("범주형 컬럼이 없습니다.")
        return

    for column in columns:
        print(f"\n- {column}")
        print(
            dataframe.group_by(column)
            .len()
            .sort("len", descending=True)
            .head(top_n)
        )


def run_basic_eda(
    pandas_df: pd.DataFrame,
    polars_df: pl.DataFrame,
    top_n: int = 5,
) -> None:
    """Pandas와 Polars DataFrame의 기본 EDA를 차례로 실행합니다."""
    if top_n <= 0:
        raise ValueError("top_n은 0보다 커야 합니다.")

    run_pandas_eda(pandas_df, top_n=top_n)
    run_polars_eda(polars_df, top_n=top_n)
