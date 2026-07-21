"""Adult Census Income 데이터 전처리 함수 모음."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# 원본 CSV는 결측치를 공백을 포함한 ' ?' 문자열로 표기한다.
MISSING_TOKEN = " ?"


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """표준 전처리를 수행하고 (정제된 DataFrame, 전처리 요약 dict)를 반환한다.

    수행 순서:
        1. 결측치 토큰(' ?') -> NaN 변환
        2. 결측 비율 계산
        3. 결측치가 있는 행 제거 (결측 비율이 낮으므로 dropna로 처리)
        4. 중복 행 제거
        5. 문자열 컬럼 앞뒤 공백 strip
        6. target(income) 을 0/1 로 변환
    """
    logger.info("전처리 시작: 입력 shape=%s", df.shape)
    summary: dict = {}

    # 1) 결측치 토큰(' ?') -> NaN 변환 (로딩 단계에서는 원본 그대로 유지)
    df = df.replace(MISSING_TOKEN, pd.NA)

    # 2) 결측 비율 계산: 낮은 비율(약 7%)이므로 삭제(dropna) 전략을 선택
    n_rows_before_na = len(df)
    rows_with_na = df.isna().any(axis=1).sum()
    missing_ratio = rows_with_na / n_rows_before_na
    summary["rows_before_dropna"] = n_rows_before_na
    summary["rows_with_missing"] = int(rows_with_na)
    summary["missing_ratio"] = round(float(missing_ratio), 4)
    logger.info(
        "결측치 계산 완료: 결측 행=%d/%d (비율=%.2f%%)",
        rows_with_na, n_rows_before_na, missing_ratio * 100,
    )

    df = df.dropna().reset_index(drop=True)
    summary["rows_after_dropna"] = len(df)
    logger.info("dropna 완료: 남은 행=%d", len(df))

    # 3) 중복 행 제거 (제거 전후 행 수 기록)
    n_before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    summary["rows_before_dedup"] = n_before_dedup
    summary["rows_after_dedup"] = len(df)
    summary["duplicates_removed"] = n_before_dedup - len(df)
    logger.info(
        "중복 제거 완료: %d -> %d (%d건 제거)",
        n_before_dedup, len(df), summary["duplicates_removed"],
    )

    # 4) 문자열 컬럼 앞뒤 공백 제거 (CSV 원본이 ", " 로 구분되어 값 앞에 공백이 남아있음)
    str_cols = df.select_dtypes(include="object").columns
    logger.info("문자열 컬럼 strip 처리: %s", list(str_cols))
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 5) target 변환: income -> 0(<=50K) / 1(>50K)
    df["income"] = df["income"].str.rstrip(".")  # 방어적 처리(.test 파일 형식 대비)
    df["income"] = (df["income"] == ">50K").astype(int)
    logger.info("target(income) 이진화 완료")

    summary["final_shape"] = df.shape
    summary["target_positive_ratio"] = round(float(df["income"].mean()), 4)
    logger.info(
        "전처리 종료: 최종 shape=%s, 양성 비율=%.2f%%",
        summary["final_shape"], summary["target_positive_ratio"] * 100,
    )

    return df, summary
