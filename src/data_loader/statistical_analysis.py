import logging
from typing import Any, Dict, Tuple

import pandas as pd
from scipy.stats import ttest_ind


logger = logging.getLogger(__name__)


def calculate_descriptive_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """숫자형 변수의 평균, 표준편차와 사분위수를 계산합니다."""
    numeric_dataframe = dataframe.select_dtypes(include="number")
    if numeric_dataframe.empty:
        raise ValueError("기술통계를 계산할 숫자형 컬럼이 없습니다.")

    return numeric_dataframe.describe().transpose()[
        ["mean", "std", "25%", "50%", "75%"]
    ]


def calculate_correlations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """숫자형 변수 사이의 Pearson 상관계수를 계산합니다."""
    numeric_dataframe = dataframe.select_dtypes(include="number")
    if numeric_dataframe.shape[1] < 2:
        raise ValueError("상관계수 계산에는 숫자형 컬럼이 2개 이상 필요합니다.")

    return numeric_dataframe.corr(method="pearson")


def perform_independent_t_test(
    dataframe: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """두 독립 그룹의 평균 차이를 Welch t-test로 검정합니다."""
    if value_column not in dataframe.columns:
        raise ValueError(f"존재하지 않는 분석 변수입니다: {value_column}")
    if group_column not in dataframe.columns:
        raise ValueError(f"존재하지 않는 그룹 변수입니다: {group_column}")
    if not 0 < alpha < 1:
        raise ValueError("alpha는 0과 1 사이여야 합니다.")

    first_group = dataframe.loc[
        dataframe[group_column] == group_a,
        value_column,
    ].dropna()
    second_group = dataframe.loc[
        dataframe[group_column] == group_b,
        value_column,
    ].dropna()

    if len(first_group) < 2 or len(second_group) < 2:
        raise ValueError("t-test를 위해 각 그룹에 2개 이상의 값이 필요합니다.")

    result = ttest_ind(
        first_group,
        second_group,
        equal_var=False,
        nan_policy="omit",
    )
    p_value = float(result.pvalue)
    reject_null = p_value < alpha

    if reject_null:
        interpretation = (
            f"p-value가 {alpha}보다 작으므로 귀무가설을 기각합니다. "
            "두 그룹의 평균에는 통계적으로 유의한 차이가 있습니다."
        )
    else:
        interpretation = (
            f"p-value가 {alpha} 이상이므로 귀무가설을 기각할 수 없습니다. "
            "두 그룹의 평균 차이가 통계적으로 유의하다고 보기 어렵습니다."
        )

    return {
        "value_column": value_column,
        "group_column": group_column,
        "group_a": group_a,
        "group_b": group_b,
        "group_a_count": len(first_group),
        "group_b_count": len(second_group),
        "group_a_mean": float(first_group.mean()),
        "group_b_mean": float(second_group.mean()),
        "t_statistic": float(result.statistic),
        "p_value": p_value,
        "alpha": alpha,
        "reject_null": reject_null,
        "interpretation": interpretation,
    }


def run_statistical_analysis(
    dataframe: pd.DataFrame,
    value_column: str = "hours-per-week",
    group_column: str = "income",
    groups: Tuple[str, str] = ("<=50K", ">50K"),
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """기술통계, 상관분석과 독립표본 t-test를 실행하고 출력합니다."""
    descriptive = calculate_descriptive_statistics(dataframe)
    correlations = calculate_correlations(dataframe)
    t_test = perform_independent_t_test(
        dataframe=dataframe,
        value_column=value_column,
        group_column=group_column,
        group_a=groups[0],
        group_b=groups[1],
        alpha=alpha,
    )

    print("\n========== 통계 분석 ==========")
    print("\n[기술통계: 평균·표준편차·분위수]")
    print(descriptive)

    print("\n[Pearson 상관계수]")
    print(correlations)

    print("\n[독립표본 t-test: Welch 방식]")
    print(
        f"{groups[0]}: n={t_test['group_a_count']}, "
        f"평균={t_test['group_a_mean']:.4f}"
    )
    print(
        f"{groups[1]}: n={t_test['group_b_count']}, "
        f"평균={t_test['group_b_mean']:.4f}"
    )
    print(f"t-statistic: {t_test['t_statistic']:.6f}")
    print(f"p-value: {t_test['p_value']:.6g}")
    print(f"해석: {t_test['interpretation']}")

    logger.info(
        "통계 분석 완료: variable=%s, groups=%s, p_value=%s",
        value_column,
        groups,
        t_test["p_value"],
    )
    return {
        "descriptive_statistics": descriptive,
        "correlations": correlations,
        "t_test": t_test,
    }
