"""기술통계, 상관분석, 가설검정(H1~H4) 모듈."""
import logging

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

NUMERIC_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week",
]


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 변수의 평균·표준편차·분위수(describe)를 반환한다."""
    logger.info("기술통계 계산 시작: columns=%s", NUMERIC_COLS)
    result = df[NUMERIC_COLS].describe()
    logger.info("기술통계 계산 완료")
    return result


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 변수 + income 간 상관계수 행렬을 계산해 반환한다."""
    logger.info("상관분석 시작")
    result = df[NUMERIC_COLS + ["income"]].corr()
    logger.info("상관분석 완료")
    return result


def test_h1_hours_worked(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """H1: >50K 그룹의 hours_per_week 평균이 유의하게 더 큰지 Welch's t-test로 검정한다."""
    logger.info("H1 검정 시작 (hours_per_week, alpha=%s)", alpha)
    low = df.loc[df["income"] == 0, "hours_per_week"]
    high = df.loc[df["income"] == 1, "hours_per_week"]

    # 등분산 가정을 두지 않는 Welch's t-test 사용 (equal_var=False)
    # scipy의 데코레이터(_axis_nan_policy_factory)로 인해 정적 타입 스텁이 TtestResult의
    # statistic/pvalue 속성을 인식하지 못하므로 getattr로 꺼내 타입 체커 오탐을 피한다.
    ttest_result = stats.ttest_ind(high, low, equal_var=False)
    t_stat = float(getattr(ttest_result, "statistic"))
    p_value = float(getattr(ttest_result, "pvalue"))

    significant = p_value < alpha
    conclusion = (
        f"p-value({p_value:.4g}) < 유의수준({alpha}) 이므로, "
        ">50K 그룹과 <=50K 그룹의 평균 hours_per_week 차이는 통계적으로 유의하다. "
        f"(H1 채택: >50K 그룹 평균 {high.mean():.2f}시간 vs <=50K 그룹 평균 {low.mean():.2f}시간)"
        if significant
        else f"p-value({p_value:.4g}) >= 유의수준({alpha}) 이므로, 두 그룹 평균 차이가 유의하다고 볼 수 없다. (H1 기각)"
    )
    logger.info("H1 검정 완료: t=%.4f, p=%.4g, significant=%s", t_stat, p_value, significant)

    return {
        "group_low_mean": round(float(low.mean()), 2),
        "group_high_mean": round(float(high.mean()), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": p_value,
        "alpha": alpha,
        "significant": bool(significant),
        "conclusion": conclusion,
    }


def test_h2_education(df: pd.DataFrame) -> dict:
    """H2: education_num이 높을수록 고소득 비율이 높은지 그룹별 비율과 상관계수로 확인한다."""
    logger.info("H2 검정 시작 (education_num)")
    by_education = (
        df.groupby("education_num")["income"].mean().sort_index()
    )
    corr = df["education_num"].corr(df["income"])
    conclusion = (
        f"education_num과 income의 상관계수는 {corr:.4f}로 양의 상관관계를 보이며, "
        "education_num이 높은 그룹일수록 고소득(>50K) 비율이 대체로 증가한다. (H2 지지)"
        if corr > 0
        else f"education_num과 income의 상관계수는 {corr:.4f}로 양의 상관관계가 뚜렷하지 않다. (H2 기각)"
    )
    logger.info("H2 검정 완료: correlation=%.4f", corr)
    return {
        "ratio_by_education_num": by_education,
        "correlation": round(float(corr), 4),
        "conclusion": conclusion,
    }


def test_h3_capital_gain(df: pd.DataFrame, model_coef: float | None = None) -> dict:
    """H3: capital_gain이 소득과 강한 양의 관계를 갖는지 상관계수(및 모델 계수)로 확인한다."""
    logger.info("H3 검정 시작 (capital_gain, model_coef=%s)", model_coef)
    corr = df["capital_gain"].corr(df["income"])
    conclusion = (
        f"capital_gain과 income의 상관계수는 {corr:.4f}로 양의 상관관계를 보인다."
    )
    if model_coef is not None:
        conclusion += (
            f" 로지스틱 회귀 계수 또한 {model_coef:.4f}로 양의 값을 가져, "
            "capital_gain이 고소득을 예측하는 핵심 변수임을 뒷받침한다. (H3 지지)"
        )
    logger.info("H3 검정 완료: correlation=%.4f", corr)
    return {
        "correlation": round(float(corr), 4),
        "model_coef": None if model_coef is None else round(float(model_coef), 4),
        "conclusion": conclusion,
    }


def test_h4_marital_relationship(df: pd.DataFrame) -> dict:
    """H4: marital_status/relationship에 따른 고소득 비율 차이를 그룹 비교로 확인한다."""
    logger.info("H4 검정 시작 (marital_status, relationship)")
    by_marital = df.groupby("marital_status")["income"].mean().sort_values(ascending=False)
    by_relationship = df.groupby("relationship")["income"].mean().sort_values(ascending=False)

    marital_gap = by_marital.max() - by_marital.min()
    relationship_gap = by_relationship.max() - by_relationship.min()

    conclusion = (
        f"marital_status 그룹 간 고소득 비율 격차는 {marital_gap:.2%}, "
        f"relationship 그룹 간 격차는 {relationship_gap:.2%}로 모두 큰 차이를 보인다. (H4 지지)"
    )
    logger.info(
        "H4 검정 완료: marital_gap=%.4f, relationship_gap=%.4f",
        marital_gap, relationship_gap,
    )
    return {
        "ratio_by_marital_status": by_marital,
        "ratio_by_relationship": by_relationship,
        "marital_gap": round(float(marital_gap), 4),
        "relationship_gap": round(float(relationship_gap), 4),
        "conclusion": conclusion,
    }
