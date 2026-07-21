"""분석 결과를 종합하여 report.md 를 자동 생성하는 모듈."""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(context: dict, out_path: Path) -> Path:
    """수집된 모든 분석 결과(dict)를 바탕으로 report.md 를 작성한다."""
    logger.info("리포트 생성 시작")
    lines: list[str] = []

    lines.append("# Adult Census Income — 로지스틱 회귀 분석 리포트\n")

    # 1. 데이터 개요 & Pandas vs Polars 비교
    cmp = context["pandas_polars_comparison"]
    lines.append("## 1. 데이터 개요 및 Pandas vs Polars 로딩 비교\n")
    lines.append(f"- 원본 데이터 shape: `{cmp['pandas_shape']}`")
    lines.append(f"- Pandas 로딩 시간: `{cmp['pandas_time_sec']}초`")
    lines.append(f"- Polars 로딩 시간: `{cmp['polars_time_sec']}초`")
    lines.append("")
    lines.append("| 컬럼 | Pandas dtype | Polars dtype |")
    lines.append("| --- | --- | --- |")
    for col in cmp["pandas_dtypes"]:
        lines.append(f"| {col} | {cmp['pandas_dtypes'][col]} | {cmp['polars_dtypes'][col]} |")
    lines.append("")
    if cmp.get("notes"):
        lines.append("**주의할 차이점**\n")
        for note in cmp["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    # 2. 전처리 요약
    clean = context["clean_summary"]
    lines.append("## 2. 전처리 요약\n")
    lines.append("- 결측치 토큰(' ?') → NaN 변환 후 결측 비율 계산")
    lines.append(
        f"- 결측치가 있는 행: {clean['rows_with_missing']}/{clean['rows_before_dropna']} "
        f"(비율 {clean['missing_ratio']:.2%}) → 결측 비율이 낮아 해당 행을 dropna로 제거"
    )
    lines.append(f"- dropna 이후 행 수: {clean['rows_after_dropna']}")
    lines.append(
        f"- 중복 행 제거: {clean['rows_before_dedup']} → {clean['rows_after_dedup']} "
        f"({clean['duplicates_removed']}건 제거)"
    )
    lines.append(f"- 최종 shape: {clean['final_shape']}")
    lines.append(
        f"- target(income) 클래스 비율: >50K = {clean['target_positive_ratio']:.2%} "
        f"(불균형 데이터 — <=50K 클래스가 다수)"
    )
    lines.append("")

    # 3. EDA 요약: 기술통계 & 상관계수
    lines.append("## 3. EDA 요약\n")
    lines.append("### 3-1. 기술통계 (수치형 변수)\n")
    lines.append(context["describe_md"])
    lines.append("")
    lines.append("### 3-2. 상관계수 (수치형 변수 + income)\n")
    lines.append(context["corr_md"])
    lines.append("")

    # 4. 가설 검증 결과
    lines.append("## 4. 가설 검증 결과\n")

    h1 = context["h1"]
    lines.append("### H1. >50K 그룹의 hours_per_week 가 유의하게 더 길다 (Welch's t-test)\n")
    lines.append(f"- <=50K 그룹 평균: {h1['group_low_mean']}시간")
    lines.append(f"- >50K 그룹 평균: {h1['group_high_mean']}시간")
    lines.append(f"- t-statistic: {h1['t_statistic']}, p-value: {h1['p_value']:.4g}")
    lines.append(f"- 해석 (유의수준 {h1['alpha']}): {h1['conclusion']}")
    lines.append("")

    h2 = context["h2"]
    lines.append("### H2. education_num이 높을수록 고소득 비율이 높다\n")
    lines.append(f"- education_num과 income의 상관계수: {h2['correlation']}")
    lines.append(f"- 해석: {h2['conclusion']}")
    lines.append("")

    h3 = context["h3"]
    lines.append("### H3. capital_gain은 소득과 강한 양의 관계를 가지는 핵심 예측 변수다\n")
    lines.append(f"- capital_gain과 income의 상관계수: {h3['correlation']}")
    lines.append(f"- 로지스틱 회귀 계수: {h3['model_coef']}")
    lines.append(f"- 해석: {h3['conclusion']}")
    lines.append("")

    h4 = context["h4"]
    lines.append("### H4. marital_status/relationship에 따라 고소득 비율 차이가 크다\n")
    lines.append(f"- marital_status 그룹 간 최대-최소 격차: {h4['marital_gap']:.2%}")
    lines.append(f"- relationship 그룹 간 최대-최소 격차: {h4['relationship_gap']:.2%}")
    lines.append(f"- 해석: {h4['conclusion']}")
    lines.append("")

    # 5. 모델 성능
    lines.append("## 5. ML 모델(로지스틱 회귀) 성능\n")
    lines.append(f"- 학습 데이터: {context['n_train']}건 / 테스트 데이터: {context['n_test']}건\n")
    m = context["metrics"]
    lines.append("| 지표 | 값 |")
    lines.append("| --- | --- |")
    lines.append(f"| Accuracy | {m['accuracy']:.4f} |")
    lines.append(f"| Precision | {m['precision']:.4f} |")
    lines.append(f"| Recall | {m['recall']:.4f} |")
    lines.append(f"| F1-score | {m['f1']:.4f} |")
    lines.append(f"| ROC-AUC | {m['roc_auc']:.4f} |")
    lines.append("")
    lines.append(
        "> 주의: target 클래스가 불균형(>50K가 소수)하므로 accuracy만으로 모델 성능을 "
        "판단하지 않는다. Precision/Recall/F1/ROC-AUC를 함께 확인해야 소수 클래스 "
        "탐지력을 올바르게 평가할 수 있다.\n"
    )

    lines.append("### Confusion Matrix\n")
    lines.append("```")
    lines.append(str(context["confusion_matrix"]))
    lines.append("```\n")

    lines.append("### Classification Report\n")
    lines.append("```")
    lines.append(context["classification_report"])
    lines.append("```\n")

    lines.append("### 로지스틱 회귀 계수 (상위 양/음 피처) 및 가설 연결 해석\n")
    lines.append(context["coef_md"])
    lines.append("")
    lines.append(
        "- capital_gain 계수가 강한 양수로 나타나 **H3**을 뒷받침한다.\n"
        "- education_num 계수가 양수로 나타나 **H2**를 뒷받침한다.\n"
        "- marital_status/relationship 관련 원-핫 피처들이 상위 계수에 다수 포함되어 "
        "**H4**를 뒷받침한다.\n"
    )

    # 6. 산출물 목록
    lines.append("## 6. 산출물 목록\n")
    for name, path in context["artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("리포트 생성 완료 (%d줄)", len(lines))
    return out_path


def df_to_markdown(df: pd.DataFrame, index_label: str = "", show_index: bool = True) -> str:
    """DataFrame을 report.md에 삽입할 마크다운 테이블 문자열로 변환한다.

    (외부 의존성인 tabulate 없이 순수 pandas만으로 변환하기 위해 직접 구현한다.)
    """
    logger.debug("DataFrame -> markdown 변환: shape=%s", df.shape)
    display_df = df.round(4) if df.select_dtypes(include="number").shape[1] else df
    headers = ([index_label or ""] if show_index else []) + [str(c) for c in display_df.columns]
    rows = [
        ([str(idx)] if show_index else []) + [str(v) for v in row]
        for idx, row in zip(display_df.index, display_df.values)
    ]

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
