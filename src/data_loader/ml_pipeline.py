import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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


def build_income_pipeline(features: pd.DataFrame) -> Pipeline:
    """전처리와 로지스틱 회귀 모델을 하나의 Pipeline으로 구성합니다."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"],
    ).columns.tolist()

    if not numeric_columns and not categorical_columns:
        raise ValueError("모델 학습에 사용할 컬럼이 없습니다.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_evaluate_save_model(
    dataframe: pd.DataFrame,
    target_column: str = "income",
    model_path: str = "outputs/models/adult_income_pipeline.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Pipeline을 학습·평가하고 전처리를 포함한 모델을 저장합니다."""
    if target_column not in dataframe.columns:
        raise ValueError(f"타깃 컬럼이 존재하지 않습니다: {target_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size는 0과 1 사이여야 합니다.")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    if target.nunique() != 2:
        raise ValueError("현재 모델은 두 개 클래스의 이진 분류만 지원합니다.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    pipeline = build_income_pipeline(features)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    positive_label = ">50K"
    if positive_label not in pipeline.classes_:
        positive_label = pipeline.classes_[1]
    positive_index = list(pipeline.classes_).index(positive_label)
    probabilities = pipeline.predict_proba(x_test)[:, positive_index]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test,
            predictions,
            pos_label=positive_label,
            zero_division=0,
        )),
        "recall": float(recall_score(
            y_test,
            predictions,
            pos_label=positive_label,
            zero_division=0,
        )),
        "f1": float(f1_score(
            y_test,
            predictions,
            pos_label=positive_label,
            zero_division=0,
        )),
        "roc_auc": float(roc_auc_score(
            (y_test == positive_label).astype(int),
            probabilities,
        )),
    }
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=pipeline.classes_,
    )

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)

    print("\n========== 머신러닝 모델 평가 ==========")
    print(f"학습 데이터: {len(x_train)}건")
    print(f"평가 데이터: {len(x_test)}건")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\n[Classification Report]")
    print(report)
    print(f"[Confusion Matrix: {list(pipeline.classes_)}]")
    print(matrix)
    print(f"\n모델 저장 경로: {destination.resolve()}")

    logger.info("ML Pipeline 학습 및 저장 완료: %s", destination.resolve())
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "model_path": destination,
    }
