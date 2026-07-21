"""
============================================================
프로그램명 : clean.py
설    명   : 데이터 수집(Extract) 및 전처리(Transform) 담당 모듈.
             - Adult Census Income 원본 다운로드 및 로컬 캐시
             - Pandas / Polars 양쪽으로 로딩하여 결과 비교
             - 결측치 · 중복 처리
             - 기본 EDA 요약 산출
작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

import statistics
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import polars as pl

from src.config import (
    COLUMNS,
    DATA_URL,
    PROCESSED_FILE,
    RAW_FILE,
    TARGET,
    ensure_dirs,
    get_logger,
)

logger = get_logger("clean")


class DataLoadError(RuntimeError):
    """데이터 수집 단계에서 복구 불가능한 오류가 발생했음을 나타내는 예외."""


# ------------------------------------------------------------------
# 1. Extract — 원본 데이터 확보
# ------------------------------------------------------------------
def download_raw(url: str = DATA_URL, dest=RAW_FILE, force: bool = False):
    """원본 CSV를 내려받아 data/raw 에 저장하고 경로를 반환한다.

    이미 파일이 있으면 네트워크 호출 없이 캐시를 재사용한다.
    (원본 데이터는 수정 금지 원칙에 따라 raw 폴더에만 보관)

    Args:
        url: 원본 데이터 URL.
        dest: 저장할 로컬 경로.
        force: True 이면 캐시가 있어도 다시 내려받는다.

    Returns:
        내려받은(또는 캐시된) 파일의 Path.

    Raises:
        DataLoadError: 네트워크 오류 등으로 다운로드에 실패한 경우.
    """
    ensure_dirs()

    if dest.exists() and not force:
        logger.info("원본 캐시 사용: %s (%.1f KB)", dest, dest.stat().st_size / 1024)
        return dest

    logger.info("원본 다운로드 시작: %s", url)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - 신뢰 가능한 UCI 주소
    except (urllib.error.URLError, OSError) as exc:
        raise DataLoadError(f"데이터 다운로드 실패: {url}") from exc

    logger.info("다운로드 완료: %s (%.1f KB)", dest, dest.stat().st_size / 1024)
    return dest


def load_with_pandas(path=RAW_FILE) -> tuple[pd.DataFrame, float]:
    """Pandas로 원본을 읽고 (DataFrame, 소요시간) 을 반환한다.

    Adult 원본은 헤더가 없고 결측치가 ' ?' 로 표기되어 있으므로
    header=None / names / na_values / skipinitialspace 옵션을 지정한다.

    주의: skipinitialspace=True 가 값 앞 공백을 먼저 제거하므로
          na_values 에는 ' ?' 뿐 아니라 '?' 도 함께 넣어야 결측으로 인식된다.
    """
    start = time.perf_counter()
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?", " ?"],
        skipinitialspace=True,
        skip_blank_lines=True,
    )
    elapsed = time.perf_counter() - start
    logger.debug("Pandas 로딩 완료: shape=%s, %.3f초", df.shape, elapsed)
    return df, elapsed


def load_with_polars(path=RAW_FILE) -> tuple[pl.DataFrame, float]:
    """Polars로 동일한 원본을 읽고 (DataFrame, 소요시간) 을 반환한다.

    Pandas와 같은 결과가 나오는지 비교하기 위한 용도이며,
    Lazy API(scan_csv)로 실행 계획을 세운 뒤 collect() 시점에 실제로 읽는다.

    Pandas와 결과를 맞추기 위한 처리
      - 문자열 앞뒤 공백 제거      (Pandas skipinitialspace=True 대응)
      - 공백 제거 후 '?' 를 결측 처리 (Pandas na_values 대응)
      - 전 컬럼이 결측인 행 제거    (원본 끝 빈 줄, Pandas skip_blank_lines 대응)
    """
    start = time.perf_counter()
    df = (
        pl.scan_csv(
            path,
            has_header=False,
            new_columns=COLUMNS,
            separator=",",
            truncate_ragged_lines=True,
        )
        .with_columns(pl.col(pl.String).str.strip_chars())
        .with_columns(
            pl.when(pl.col(pl.String) == "?")
            .then(None)
            .otherwise(pl.col(pl.String))
            .name.keep()
        )
        .filter(~pl.all_horizontal(pl.all().is_null()))
        .collect()
    )
    elapsed = time.perf_counter() - start
    logger.debug("Polars 로딩 완료: shape=%s, %.3f초", df.shape, elapsed)
    return df, elapsed


def benchmark_loaders(
    path=RAW_FILE, repeat: int = 10, warmup: int = 2
) -> tuple[float, float]:
    """두 로더의 로딩 시간을 반복 측정해 (Pandas 중앙값, Polars 중앙값) 을 반환한다.

    1회만 재면 Polars 첫 호출에 스레드 풀 초기화 비용이 통째로 섞여 들어가
    실행할 때마다 두 도구의 우열이 뒤바뀐다. 그래서
      - 앞 `warmup` 회는 초기화 비용을 털어내는 용도로 버리고
      - 나머지 `repeat` 회의 중앙값을 사용한다 (평균은 이상치에 흔들림)

    Args:
        path: 측정 대상 원본 경로.
        repeat: 중앙값 계산에 사용할 측정 횟수.
        warmup: 버릴 예열 횟수.

    Returns:
        (Pandas 중앙값 초, Polars 중앙값 초)
    """
    pandas_times: list[float] = []
    polars_times: list[float] = []

    for i in range(warmup + repeat):
        _, t_pd = load_with_pandas(path)
        _, t_pl = load_with_polars(path)
        if i >= warmup:
            pandas_times.append(t_pd)
            polars_times.append(t_pl)

    med_pandas = statistics.median(pandas_times)
    med_polars = statistics.median(polars_times)
    logger.info(
        "로딩 벤치마크 | 예열 %d회 후 %d회 중앙값 — Pandas %.4fs, Polars %.4fs",
        warmup,
        repeat,
        med_pandas,
        med_polars,
    )
    return med_pandas, med_polars


def compare_loaders(
    pdf: pd.DataFrame, pldf: pl.DataFrame, t_pandas: float, t_polars: float
) -> dict[str, Any]:
    """Pandas와 Polars 로딩 결과가 동일한지 비교한 요약 dict를 반환한다.

    행 수 · 열 수 · 결측치 총합이 모두 같으면 두 도구가 같은 데이터를 읽은 것으로 본다.
    """
    summary = {
        "pandas_shape": pdf.shape,
        "polars_shape": (pldf.height, pldf.width),
        "shape_equal": pdf.shape == (pldf.height, pldf.width),
        "pandas_nulls": int(pdf.isna().sum().sum()),
        "polars_nulls": int(pldf.null_count().sum_horizontal().item()),
        "pandas_sec": round(t_pandas, 4),
        "polars_sec": round(t_polars, 4),
        "speedup": round(t_pandas / t_polars, 2) if t_polars > 0 else None,
    }
    summary["nulls_equal"] = summary["pandas_nulls"] == summary["polars_nulls"]

    logger.info(
        "로더 비교 | shape 일치=%s, 결측 일치=%s, Pandas %.3fs vs Polars %.3fs",
        summary["shape_equal"],
        summary["nulls_equal"],
        t_pandas,
        t_polars,
    )
    return summary


# ------------------------------------------------------------------
# 2. Transform — 결측치 · 중복 처리
# ------------------------------------------------------------------
def basic_eda(df: pd.DataFrame) -> dict[str, Any]:
    """기본 EDA 결과(행/열 수, 결측 현황, 타깃 분포 등)를 dict로 정리한다."""
    null_cnt = df.isna().sum()
    eda = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        # 결측치가 있는 컬럼만 (건수, 비율%) 로 정리
        "null_counts": {
            col: (int(cnt), round(cnt / len(df) * 100, 2))
            for col, cnt in null_cnt.items()
            if cnt > 0
        },
        "target_dist": df[TARGET].value_counts().to_dict() if TARGET in df else {},
        "describe": df.describe().round(2),
    }
    logger.info(
        "기본 EDA | %d행 %d열, 중복 %d행, 결측 컬럼 %d개",
        eda["n_rows"],
        eda["n_cols"],
        eda["n_duplicates"],
        len(eda["null_counts"]),
    )
    return eda


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """결측치와 중복을 처리한 DataFrame과 처리 이력을 반환한다.

    처리 전략
      1) 중복 행 제거 — 설문 데이터 특성상 완전히 동일한 행은 입력 오류로 간주
      2) 범주형 결측치 → 최빈값(mode) 대체
         (workclass/occupation/native-country 3개 컬럼에만 결측 존재,
          비율이 6% 미만이라 행 삭제보다 대체가 정보 손실이 적다)
      3) 수치형 결측치 → 중앙값(median) 대체 (이상치 영향을 덜 받음)
      4) 타깃(income) 문자열 정리 — 원본에 '.' 이 섞인 경우 대비

    Args:
        df: 원본 DataFrame.

    Returns:
        (정제된 DataFrame, {단계: 처리건수} 이력 dict)
    """
    out = df.copy()  # 원본 훼손 방지 (Pandas 2.x CoW 경고 회피)
    history: dict[str, int] = {}

    # 1) 중복 제거
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    history["dropped_duplicates"] = before - len(out)

    # 2) 범주형 결측치 → 최빈값
    cat_cols = out.select_dtypes(include="object").columns
    filled_cat = 0
    for col in cat_cols:
        missing = int(out[col].isna().sum())
        if missing == 0:
            continue
        out[col] = out[col].fillna(out[col].mode(dropna=True)[0])
        filled_cat += missing
    history["filled_categorical"] = filled_cat

    # 3) 수치형 결측치 → 중앙값
    num_cols = out.select_dtypes(include="number").columns
    filled_num = 0
    for col in num_cols:
        missing = int(out[col].isna().sum())
        if missing == 0:
            continue
        out[col] = out[col].fillna(out[col].median())
        filled_num += missing
    history["filled_numeric"] = filled_num

    # 4) 타깃 라벨 정리 (' >50K.' → '>50K')
    if TARGET in out.columns:
        out[TARGET] = out[TARGET].astype(str).str.strip().str.rstrip(".")

    logger.info(
        "전처리 완료 | 중복 %d행 제거, 범주형 %d건·수치형 %d건 대체, 최종 %d행",
        history["dropped_duplicates"],
        filled_cat,
        filled_num,
        len(out),
    )
    return out, history


def save_processed(df: pd.DataFrame, path=PROCESSED_FILE):
    """전처리 완료 데이터를 Parquet으로 저장한다(CSV 대비 용량·속도 이점)."""
    ensure_dirs()
    df.to_parquet(path, index=False)
    logger.info("전처리 데이터 저장: %s", path)
    return path
