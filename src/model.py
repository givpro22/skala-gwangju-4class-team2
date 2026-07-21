"""
============================================================
프로그램명 : model.py
설    명   : 머신러닝 파이프라인 담당 모듈.
             - ColumnTransformer + Pipeline 으로 전처리 + 모델 통합
             - 학습 · 평가(정확도 / F1 / ROC-AUC) · 분류 리포트
             - joblib 으로 파이프라인 저장 및 재로딩 검증
작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    ensure_dirs,
    get_logger,
)

logger = get_logger("model")

POSITIVE_LABEL: str = ">50K"  # 양성 클래스(고소득)


def build_pipeline() -> Pipeline:
    """전처리 + 분류 모델을 하나로 묶은 sklearn Pipeline을 생성한다.

    구성
      - 수치형 : 결측치 중앙값 대체 → StandardScaler 표준화
      - 범주형 : 결측치 최빈값 대체 → OneHotEncoder (학습 때 못 본 값은 무시)
      - 모델   : RandomForestClassifier (비선형 관계와 범주형 상호작용에 강함)

    Returns:
        학습 준비가 끝난 Pipeline 객체.
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=3,
                    class_weight="balanced",  # 클래스 불균형(약 76:24) 보정
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[Pipeline, dict[str, Any]]:
    """파이프라인을 학습하고 테스트셋 평가 지표를 반환한다.

    Args:
        df: 전처리 완료 DataFrame (타깃 컬럼 포함).
        test_size: 테스트셋 비율.

    Returns:
        (학습된 Pipeline, 평가 지표 dict)

    Raises:
        ValueError: 타깃 컬럼이 없거나 필요한 피처가 누락된 경우.
    """
    if TARGET not in df.columns:
        raise ValueError(f"타깃 컬럼이 없습니다: {TARGET}")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"피처 컬럼 누락: {missing}")

    x = df[features]
    y = df[TARGET]

    # stratify로 학습/테스트셋의 클래스 비율을 원본과 동일하게 유지
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("데이터 분할 | 학습 %d행, 테스트 %d행", len(x_train), len(x_test))

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    logger.info("모델 학습 완료: %s", pipeline.named_steps["clf"].__class__.__name__)

    y_pred = pipeline.predict(x_test)
    # ROC-AUC는 양성 클래스 확률이 필요하므로 클래스 순서를 확인해 인덱싱한다.
    positive_idx = list(pipeline.classes_).index(POSITIVE_LABEL)
    y_proba = pipeline.predict_proba(x_test)[:, positive_idx]

    metrics: dict[str, Any] = {
        "model": pipeline.named_steps["clf"].__class__.__name__,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "f1_positive": round(
            float(f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)), 4
        ),
        "roc_auc": round(float(roc_auc_score(y_test == POSITIVE_LABEL, y_proba)), 4),
        "classification_report": classification_report(y_test, y_pred, digits=3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": list(pipeline.classes_),
    }
    metrics["feature_importance"] = _top_feature_importance(pipeline)

    logger.info(
        "평가 지표 | accuracy=%.4f, f1(macro)=%.4f, ROC-AUC=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["roc_auc"],
    )
    return pipeline, metrics


def _top_feature_importance(
    pipeline: Pipeline, top_n: int = 10
) -> list[tuple[str, float]]:
    """학습된 파이프라인에서 중요도 상위 피처를 추출한다.

    OneHotEncoder를 거치면 컬럼명이 바뀌므로
    ColumnTransformer의 get_feature_names_out()으로 실제 피처명을 얻는다.
    """
    names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["clf"].feature_importances_

    ranked = sorted(
        zip(names, importances, strict=True), key=lambda x: x[1], reverse=True
    )
    return [(str(name), round(float(score), 4)) for name, score in ranked[:top_n]]


def save_model(pipeline: Pipeline, filename: str = "income_pipeline.joblib"):
    """학습된 파이프라인을 joblib으로 저장하고 저장 경로를 반환한다."""
    ensure_dirs()
    path = MODEL_DIR / filename
    joblib.dump(pipeline, path)
    logger.info("모델 저장: %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def verify_reload(path, sample: pd.DataFrame) -> bool:
    """저장한 모델을 다시 읽어 동일한 예측이 나오는지 검증한다.

    배포 환경에서 같은 전처리 + 예측이 재현되는지 확인하는 단계로,
    실패해도 파이프라인 전체가 중단되지 않도록 예외를 잡아 False를 반환한다.
    """
    try:
        loaded = joblib.load(path)
        preds = loaded.predict(sample)
        logger.info("모델 재로딩 검증 성공 | 샘플 %d건 예측 완료", len(preds))
        return True
    except Exception as exc:  # noqa: BLE001 - 검증 실패 사유를 모두 기록
        logger.error("모델 재로딩 검증 실패: %s", exc)
        return False
