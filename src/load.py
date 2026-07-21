"""UCI Adult Census Income 데이터 수집 및 로딩 모듈."""
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

# 원본 CSV는 결측치를 공백을 포함한 ' ?' 문자열로 표기한다.
MISSING_TOKEN = " ?"


def fetch_raw_data(raw_dir: Path, url: str = DATA_URL) -> Path:
    """원본 데이터를 다운로드하여 data/raw/adult.data 로 저장한다.

    이미 파일이 존재하면 재다운로드하지 않는다(raw 데이터는 절대 수정 금지 원칙).
    """
    logger.info("raw 데이터 확인 시작")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "adult.data"
    if not raw_path.exists():
        logger.info("raw 데이터 없음 -> 다운로드 시작: url=%s", url)
        urllib.request.urlretrieve(url, raw_path)
        logger.info("다운로드 완료")
    else:
        logger.info("raw 데이터가 이미 존재하여 다운로드를 건너뜀")
    return raw_path


def load_with_pandas(raw_path: Path) -> tuple[pd.DataFrame, float]:
    """Pandas로 원본 CSV를 로딩하고 (DataFrame, 소요시간[초])를 반환한다."""
    logger.info("Pandas 로딩 시작")
    start = time.perf_counter()
    df = pd.read_csv(
        raw_path,
        header=None,
        names=COLUMNS,
        na_values=MISSING_TOKEN,
        skipinitialspace=False,
    )
    elapsed = time.perf_counter() - start
    logger.info("Pandas 로딩 완료: shape=%s, elapsed=%.4fs", df.shape, elapsed)
    return df, elapsed


def load_with_polars(raw_path: Path) -> tuple[pl.DataFrame, float]:
    """Polars로 원본 CSV를 로딩하고 (DataFrame, 소요시간[초])를 반환한다."""
    logger.info("Polars 로딩 시작")
    start = time.perf_counter()
    df = pl.read_csv(
        raw_path,
        has_header=False,
        new_columns=COLUMNS,
        null_values=MISSING_TOKEN,
        infer_schema_length=10000,
    )
    elapsed = time.perf_counter() - start
    logger.info("Polars 로딩 완료: shape=%s, elapsed=%.4fs", df.shape, elapsed)
    return df, elapsed


def compare_pandas_polars(
    pdf: pd.DataFrame, pldf: pl.DataFrame, pd_time: float, pl_time: float
) -> dict:
    """Pandas/Polars 로딩 결과(시간, shape, dtype)를 비교해 dict로 반환하고 로그로 출력한다."""
    logger.info("Pandas vs Polars 로딩 비교 시작")
    comparison = {
        "pandas_time_sec": round(pd_time, 4),
        "polars_time_sec": round(pl_time, 4),
        "pandas_shape": pdf.shape,
        "polars_shape": pldf.shape,
        "pandas_dtypes": pdf.dtypes.astype(str).to_dict(),
        "polars_dtypes": {c: str(t) for c, t in zip(pldf.columns, pldf.dtypes)},
    }

    logger.info("Pandas  : %.4fs, shape=%s", pd_time, pdf.shape)
    logger.info("Polars  : %.4fs, shape=%s", pl_time, pldf.shape)
    faster = "Polars" if pl_time < pd_time else "Pandas"
    logger.info("이번 실행에서는 %s 가 더 빠르게 로딩했다.", faster)

    notes = []
    if pdf.shape[0] != pldf.shape[0]:
        notes.append(
            "원본 파일 끝에 빈 줄이 하나 있다. Pandas는 기본값(skip_blank_lines=True)으로 "
            "이를 건너뛰지만, Polars는 이를 전 컬럼이 null인 행으로 읽어들여 행 수가 1 더 많다."
        )
    numeric_as_string = [
        c for c in COLUMNS[:-1]
        if pdf[c].dtype != "object" and comparison["polars_dtypes"].get(c) in ("String", "str")
    ]
    if numeric_as_string:
        notes.append(
            f"{numeric_as_string} 컬럼은 CSV 값 앞에 공백(', 77516' 등)이 있어 "
            "Polars의 기본 스키마 추론에서는 숫자로 인식되지 못하고 String으로 로딩된다. "
            "반면 Pandas의 C 파서는 숫자 파싱 시 앞뒤 공백을 자동으로 무시해 int64로 인식한다."
        )
    if notes:
        logger.info("주의할 차이점 %d건 발견", len(notes))
        for note in notes:
            logger.info("  - %s", note)
    comparison["notes"] = notes

    logger.info("Pandas vs Polars 로딩 비교 완료")
    return comparison
