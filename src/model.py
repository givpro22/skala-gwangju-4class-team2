"""로지스틱 회귀 기반 ML Pipeline: 학습, 평가, 계수 해석, 저장."""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# fnlwgt(census 표본 가중치)는 소득과 인과적으로 무관한 표본추출 변수이므로 예측 피처에서 제외한다.
# education(범주형)은 education_num(서수형)과 1:1로 대응되는 중복 정보이므로 education_num만 사용한다.
NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = [
    "workclass", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country",
]
TARGET = "income"


def build_pipeline() -> Pipeline:
    """수치형(StandardScaler) + 범주형(OneHotEncoder) 전처리와 로지스틱 회귀를 결합한 Pipeline을 생성한다."""
    logger.info(
        "ML 파이프라인 구성: numeric=%s, categorical=%s",
        NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """train/test 분할, 학습, 평가지표 산출, 계수 해석까지 수행하고 결과 dict를 반환한다."""
    logger.info("ML 학습/평가 시작: 전체 데이터 shape=%s", df.shape)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("train/test 분할 완료: n_train=%d, n_test=%d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    logger.info("모델 학습 시작")
    pipeline.fit(X_train, y_train)
    logger.info("모델 학습 완료")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 클래스 불균형(양성 비율이 낮음) 데이터이므로 accuracy만으로는 소수 클래스(>50K) 탐지력을
    # 판단할 수 없다. precision/recall/f1/ROC-AUC를 함께 확인해야 한다.
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    logger.info("평가지표 산출 완료: %s", metrics)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])

    coef_df = extract_top_coefficients(pipeline)

    logger.info("ML 학습/평가 종료")
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "confusion_matrix": conf_matrix,
        "classification_report": report_text,
        "coef_df": coef_df,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def extract_top_coefficients(pipeline: Pipeline, top_n: int = 10) -> pd.DataFrame:
    """OneHotEncoder로 확장된 피처명과 로지스틱 회귀 계수를 매핑해 상위 양/음 계수를 반환한다."""
    logger.info("상위 계수 추출 시작: top_n=%d", top_n)
    preprocessor = pipeline.named_steps["preprocessor"]
    ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    all_feature_names = np.concatenate([NUMERIC_FEATURES, ohe_feature_names])

    coefs = pipeline.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({"feature": all_feature_names, "coefficient": coefs})
    coef_df = coef_df.sort_values("coefficient", ascending=False)

    top_positive = coef_df.head(top_n).assign(direction="positive")
    top_negative = coef_df.tail(top_n).sort_values("coefficient").assign(direction="negative")
    result = pd.concat([top_positive, top_negative], ignore_index=True)
    logger.info("상위 계수 추출 완료: %d개 피처 반환", len(result))
    return result[["direction", "feature", "coefficient"]]


def get_capital_gain_coefficient(coef_df: pd.DataFrame) -> float | None:
    """H3 해석용: capital_gain의 로지스틱 회귀 계수를 조회한다."""
    flat = coef_df.reset_index(drop=True)
    row = flat.loc[flat["feature"] == "capital_gain"]
    coef = float(row["coefficient"].iloc[0]) if not row.empty else None
    logger.info("capital_gain 계수 조회 결과: %s", coef)
    return coef


def save_model(pipeline: Pipeline, out_path: Path) -> Path:
    """학습된 파이프라인을 joblib으로 직렬화하여 저장한다."""
    logger.info("모델 저장 시작")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    logger.info("모델 저장 완료")
    return out_path
