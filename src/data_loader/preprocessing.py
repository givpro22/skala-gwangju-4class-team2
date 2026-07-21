import logging
from typing import Tuple

import pandas as pd
import polars as pl


logger = logging.getLogger(__name__)


def clean_pandas_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """문자열을 정리하고 결측 행과 중복 행을 제거합니다."""
    cleaned = dataframe.copy()
    string_columns = cleaned.select_dtypes(
        include=["object", "string"],
    ).columns

    for column in string_columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    cleaned[string_columns] = cleaned[string_columns].replace(
        {"?": pd.NA, "": pd.NA}
    )

    before_count = len(cleaned)
    cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)

    logger.info(
        "Pandas 전처리 완료: 기존=%s, 제거=%s, 결과=%s",
        before_count,
        before_count - len(cleaned),
        len(cleaned),
    )
    return cleaned


def clean_polars_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    """문자열을 정리하고 결측 행과 중복 행을 제거합니다."""
    string_columns = [
        column
        for column, dtype in dataframe.schema.items()
        if dtype == pl.String
    ]
    cleaned = dataframe.with_columns(
        pl.col(column).str.strip_chars().alias(column)
        for column in string_columns
    )
    cleaned = cleaned.with_columns(
        pl.when(pl.col(column).is_in(["?", ""]))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column in string_columns
    )

    before_count = cleaned.height
    cleaned = cleaned.drop_nulls().unique(maintain_order=True)

    logger.info(
        "Polars 전처리 완료: 기존=%s, 제거=%s, 결과=%s",
        before_count,
        before_count - cleaned.height,
        cleaned.height,
    )
    return cleaned


def clean_dataframes(
    pandas_df: pd.DataFrame,
    polars_df: pl.DataFrame,
) -> Tuple[pd.DataFrame, pl.DataFrame]:
    """Pandas와 Polars DataFrame을 동일한 규칙으로 정리합니다."""
    return (
        clean_pandas_dataframe(pandas_df),
        clean_polars_dataframe(polars_df),
    )
