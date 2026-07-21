"""
============================================================
프로그램명 : stats_analysis.py
설    명   : 통계 분석 담당 모듈.
             - 기술통계(평균 · 표준편차 · 분위수)
             - 변수 간 상관계수
             - 독립표본 t-test 및 p-value 해석
             - 카이제곱 독립성 검정 및 p-value 해석
작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from src.config import TARGET, get_logger

logger = get_logger("stats")

ALPHA: float = 0.05  # 유의수준 5%


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 컬럼의 기술통계(개수·평균·표준편차·사분위수 등)를 반환한다."""
    desc = df.select_dtypes("number").describe().T.round(2)
    logger.info("기술통계 산출 완료: %d개 수치형 컬럼", len(desc))
    return desc


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 변수 간 피어슨 상관계수 행렬을 반환한다."""
    corr = df.select_dtypes("number").corr().round(3)
    logger.info("상관행렬 산출 완료: %dx%d", *corr.shape)
    return corr


def top_correlations(
    corr: pd.DataFrame, top_n: int = 5
) -> list[tuple[str, str, float]]:
    """상관계수 절댓값이 큰 변수 쌍을 top_n개 뽑아 반환한다.

    대각선(자기 자신)과 중복 쌍(A-B, B-A)을 제외하기 위해
    상삼각 행렬만 사용한다.
    """
    pairs: list[tuple[str, str, float]] = []
    cols = corr.columns
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            pairs.append((col_a, col_b, float(corr.loc[col_a, col_b])))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_n]


def format_p(p_value: float) -> str:
    """p-value를 읽기 좋은 문자열로 변환한다.

    p가 너무 작아 부동소수점 언더플로로 0.0 이 되는 경우가 있어,
    그대로 'p=0' 이라 쓰지 않고 표현 가능한 하한을 명시한다.
    """
    if p_value == 0.0:
        return "< 1e-308"
    return f"{p_value:.4g}"


def _interpret(p_value: float, hypothesis: str) -> str:
    """p-value를 유의수준과 비교해 사람이 읽을 수 있는 해석 문장을 만든다."""
    p_str = format_p(p_value)
    if p_value < ALPHA:
        return f"p={p_str} < {ALPHA} → 귀무가설 기각. {hypothesis}(통계적으로 유의)."
    return f"p={p_str} >= {ALPHA} → 귀무가설 기각 실패. {hypothesis} 근거 부족(우연일 수 있음)."


def run_ttest(
    df: pd.DataFrame, value_col: str = "hours-per-week", group_col: str = TARGET
) -> dict[str, Any]:
    """두 소득 그룹의 평균 차이를 독립표본 t-test로 검정한다.

    귀무가설(H0): 두 그룹의 value_col 평균은 같다.
    대립가설(H1): 두 그룹의 value_col 평균은 다르다.

    두 그룹의 분산이 다를 수 있으므로 Welch's t-test(equal_var=False)를 사용한다.

    Args:
        df: 분석 대상 DataFrame.
        value_col: 평균을 비교할 수치형 컬럼.
        group_col: 두 개의 그룹으로 나눌 범주형 컬럼.

    Returns:
        검정 통계량 · p-value · 해석 문장을 담은 dict.

    Raises:
        ValueError: 그룹이 정확히 2개가 아니거나 컬럼이 존재하지 않는 경우.
    """
    for col in (value_col, group_col):
        if col not in df.columns:
            raise ValueError(f"컬럼이 존재하지 않습니다: {col}")

    groups = sorted(df[group_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError(f"t-test는 2개 그룹만 지원합니다 (현재 {len(groups)}개)")

    sample_a = df.loc[df[group_col] == groups[0], value_col].dropna()
    sample_b = df.loc[df[group_col] == groups[1], value_col].dropna()

    t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    result = {
        "test": "Welch's independent t-test",
        "value_col": value_col,
        "group_col": group_col,
        "group_a": str(groups[0]),
        "group_b": str(groups[1]),
        "mean_a": round(float(sample_a.mean()), 3),
        "mean_b": round(float(sample_b.mean()), 3),
        "n_a": int(len(sample_a)),
        "n_b": int(len(sample_b)),
        "t_stat": round(float(t_stat), 4),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
    }
    result["interpretation"] = _interpret(
        result["p_value"],
        f"'{groups[0]}' 그룹과 '{groups[1]}' 그룹의 {value_col} 평균에 차이가 있다",
    )

    logger.info(
        "t-test | %s: %s(%.2f) vs %s(%.2f), t=%.3f, %s",
        value_col,
        groups[0],
        result["mean_a"],
        groups[1],
        result["mean_b"],
        result["t_stat"],
        result["interpretation"],
    )
    return result


def run_chi2(
    df: pd.DataFrame, col_a: str = "sex", col_b: str = TARGET
) -> dict[str, Any]:
    """두 범주형 변수의 독립성을 카이제곱 검정으로 확인한다.

    귀무가설(H0): 두 변수는 서로 독립이다.
    대립가설(H1): 두 변수는 서로 연관이 있다.

    Args:
        df: 분석 대상 DataFrame.
        col_a, col_b: 분할표를 만들 두 범주형 컬럼.

    Returns:
        카이제곱 통계량 · 자유도 · p-value · 해석 문장을 담은 dict.

    Raises:
        ValueError: 컬럼이 존재하지 않는 경우.
    """
    for col in (col_a, col_b):
        if col not in df.columns:
            raise ValueError(f"컬럼이 존재하지 않습니다: {col}")

    contingency = pd.crosstab(df[col_a], df[col_b])
    chi2, p_value, dof, _expected = stats.chi2_contingency(contingency)

    result = {
        "test": "Chi-square test of independence",
        "col_a": col_a,
        "col_b": col_b,
        "chi2": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
        "contingency": contingency,
    }
    result["interpretation"] = _interpret(
        result["p_value"], f"'{col_a}'와(과) '{col_b}'는 서로 연관이 있다"
    )

    logger.info(
        "chi2 | %s x %s: chi2=%.3f, dof=%d, %s",
        col_a,
        col_b,
        result["chi2"],
        dof,
        result["interpretation"],
    )
    return result
