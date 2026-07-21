import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    table = dataframe.reset_index()
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _relative_link(target: Path, report_path: Path) -> str:
    return Path(os.path.relpath(target, report_path.parent)).as_posix()


def generate_markdown_report(
    dataframe: pd.DataFrame,
    statistical_results: Dict[str, Any],
    ml_results: Dict[str, Any],
    seaborn_files: List[Path],
    plotly_files: List[Path],
    original_shape: Optional[Tuple[int, int]] = None,
    report_path: str = "outputs/report.md",
) -> Path:
    """분석·통계·모델·시각화 결과를 Markdown 보고서로 저장합니다."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    cleaned_shape = dataframe.shape
    source_shape = original_shape or cleaned_shape
    removed_rows = source_shape[0] - cleaned_shape[0]
    target_counts = dataframe["income"].value_counts().rename("count").to_frame()
    descriptive = statistical_results["descriptive_statistics"]
    correlations = statistical_results["correlations"]
    t_test = statistical_results["t_test"]
    metrics = ml_results["metrics"]
    matrix = pd.DataFrame(
        ml_results["confusion_matrix"],
        index=ml_results["pipeline"].classes_,
        columns=ml_results["pipeline"].classes_,
    )

    model_path = Path(ml_results["model_path"])
    model_link = _relative_link(model_path, destination)
    static_chart_lines = [
        f"### {path.stem}\n\n![{path.stem}]({_relative_link(path, destination)})"
        for path in seaborn_files
    ]
    interactive_chart_lines = [
        f"- [{path.stem}]({_relative_link(path, destination)})"
        for path in plotly_files
    ]

    report = f"""# Adult Income 데이터 분석 보고서

생성 시각: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}

## 1. 데이터 및 전처리 개요

- 원본 데이터: {source_shape[0]:,}행 × {source_shape[1]}열
- 전처리 데이터: {cleaned_shape[0]:,}행 × {cleaned_shape[1]}열
- 결측치·중복 처리로 제거된 행: {removed_rows:,}개
- 전처리 후 결측값: {int(dataframe.isna().sum().sum()):,}개
- 전처리 후 중복 행: {int(dataframe.duplicated().sum()):,}개

### 소득 클래스 분포

{_dataframe_to_markdown(target_counts)}

## 2. 기술통계

숫자형 변수의 평균, 표준편차와 사분위수입니다.

{_dataframe_to_markdown(descriptive)}

## 3. 변수 간 Pearson 상관계수

상관계수는 -1에서 1 사이이며, 절댓값이 클수록 두 변수의 선형 관계가 강합니다.

{_dataframe_to_markdown(correlations)}

## 4. 독립표본 t-test

- 분석 변수: `{t_test['value_column']}`
- 비교 그룹: `{t_test['group_a']}` vs `{t_test['group_b']}`
- {t_test['group_a']} 표본 수·평균: {t_test['group_a_count']:,}개, {t_test['group_a_mean']:.4f}
- {t_test['group_b']} 표본 수·평균: {t_test['group_b_count']:,}개, {t_test['group_b_mean']:.4f}
- t-statistic: {t_test['t_statistic']:.6f}
- p-value: {t_test['p_value']:.6g}
- 유의수준: {t_test['alpha']}
- 해석: {t_test['interpretation']}

## 5. 머신러닝 모델 평가

전처리와 `LogisticRegression`을 하나의 sklearn Pipeline으로 구성했습니다.

| 지표 | 값 |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-score | {metrics['f1']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |

### 혼동행렬

행은 실제 클래스, 열은 예측 클래스입니다.

{_dataframe_to_markdown(matrix)}

### 저장 모델

- [adult_income_pipeline.joblib]({model_link})

## 6. Seaborn 정적 시각화

{chr(10).join(static_chart_lines)}

## 7. Plotly 인터랙티브 시각화

HTML 파일을 브라우저로 열어 확대, 축소와 마우스 오버 기능을 사용할 수 있습니다.

{chr(10).join(interactive_chart_lines)}
"""
    destination.write_text(report, encoding="utf-8")
    logger.info("Markdown 분석 보고서 생성 완료: %s", destination.resolve())
    return destination
