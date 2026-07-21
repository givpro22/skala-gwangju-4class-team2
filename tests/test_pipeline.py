"""
============================================================
프로그램명 : test_pipeline.py
설    명   : 파이프라인 각 단계의 단위 테스트.
             네트워크 없이 동작하도록 fixture로 소규모 가짜 데이터를 만들어 검증한다.
실행 방법  : pytest tests/ -v
작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clean, model, stats_analysis
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """결측치·중복이 포함된 소규모 테스트용 DataFrame을 만든다.

    실제 Adult 데이터와 동일한 컬럼 구성을 유지해
    전처리·모델 함수가 그대로 동작하는지 확인할 수 있게 한다.
    """
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame(
        {
            "age": rng.integers(20, 65, n),
            "workclass": rng.choice(["Private", "Self-emp", None], n),
            "fnlwgt": rng.integers(10_000, 300_000, n),
            "education": rng.choice(["Bachelors", "HS-grad"], n),
            "education-num": rng.integers(6, 16, n),
            "marital-status": rng.choice(["Married", "Never-married"], n),
            "occupation": rng.choice(["Sales", "Tech-support", None], n),
            "relationship": rng.choice(["Husband", "Not-in-family"], n),
            "race": rng.choice(["White", "Black"], n),
            "sex": rng.choice(["Male", "Female"], n),
            "capital-gain": rng.integers(0, 5000, n),
            "capital-loss": rng.integers(0, 2000, n),
            "hours-per-week": rng.integers(20, 60, n),
            "native-country": rng.choice(["United-States", "Mexico"], n),
            TARGET: rng.choice(["<=50K", ">50K"], n),
        }
    )
    # 수치형 결측치와 중복 행을 의도적으로 삽입
    df.loc[0, "age"] = np.nan
    return pd.concat([df, df.iloc[:5]], ignore_index=True)


def test_clean_data_removes_duplicates_and_nulls(sample_df: pd.DataFrame) -> None:
    """전처리 후 중복과 결측치가 모두 사라지는지 검증한다."""
    cleaned, history = clean.clean_data(sample_df)

    assert cleaned.duplicated().sum() == 0, "중복 행이 남아 있으면 안 된다"
    assert cleaned.isna().sum().sum() == 0, "결측치가 남아 있으면 안 된다"
    assert history["dropped_duplicates"] >= 5
    assert history["filled_numeric"] >= 1


def test_clean_data_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    """전처리 함수가 원본 DataFrame을 변경하지 않는지 확인한다."""
    before_nulls = int(sample_df.isna().sum().sum())
    clean.clean_data(sample_df)
    assert int(sample_df.isna().sum().sum()) == before_nulls


def test_basic_eda_keys(sample_df: pd.DataFrame) -> None:
    """EDA 요약 dict가 필요한 키를 모두 담고 있는지 확인한다."""
    eda = clean.basic_eda(sample_df)
    for key in ("n_rows", "n_cols", "n_duplicates", "null_counts", "target_dist"):
        assert key in eda


def test_run_ttest_returns_interpretation(sample_df: pd.DataFrame) -> None:
    """t-test 결과에 p-value와 해석 문장이 포함되는지 확인한다."""
    cleaned, _ = clean.clean_data(sample_df)
    result = stats_analysis.run_ttest(cleaned)

    assert 0.0 <= result["p_value"] <= 1.0
    assert isinstance(result["significant"], bool)
    assert "p=" in result["interpretation"]


def test_run_ttest_raises_on_missing_column(sample_df: pd.DataFrame) -> None:
    """존재하지 않는 컬럼을 넘기면 ValueError가 발생해야 한다."""
    with pytest.raises(ValueError, match="컬럼이 존재하지 않습니다"):
        stats_analysis.run_ttest(sample_df, value_col="not_exist")


def test_run_chi2_contingency_shape(sample_df: pd.DataFrame) -> None:
    """카이제곱 검정의 분할표가 2x2로 만들어지는지 확인한다."""
    cleaned, _ = clean.clean_data(sample_df)
    result = stats_analysis.run_chi2(cleaned, col_a="sex", col_b=TARGET)

    assert result["contingency"].shape == (2, 2)
    assert result["dof"] == 1


def test_pipeline_fit_predict(sample_df: pd.DataFrame) -> None:
    """Pipeline이 학습되고 입력 행 수만큼 예측을 반환하는지 확인한다."""
    cleaned, _ = clean.clean_data(sample_df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    pipeline = model.build_pipeline()
    pipeline.fit(cleaned[features], cleaned[TARGET])
    preds = pipeline.predict(cleaned[features].head(10))

    assert len(preds) == 10
    assert set(preds).issubset({"<=50K", ">50K"})


def test_train_and_evaluate_raises_without_target(sample_df: pd.DataFrame) -> None:
    """타깃 컬럼이 없으면 ValueError가 발생해야 한다."""
    with pytest.raises(ValueError, match="타깃 컬럼이 없습니다"):
        model.train_and_evaluate(sample_df.drop(columns=[TARGET]))


def test_save_and_reload_model(sample_df: pd.DataFrame, tmp_path) -> None:
    """joblib 저장 후 재로딩한 모델이 동일하게 예측하는지 확인한다."""
    cleaned, _ = clean.clean_data(sample_df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    pipeline = model.build_pipeline()
    pipeline.fit(cleaned[features], cleaned[TARGET])

    path = tmp_path / "test_model.joblib"
    import joblib

    joblib.dump(pipeline, path)

    assert model.verify_reload(path, cleaned[features].head(3)) is True
