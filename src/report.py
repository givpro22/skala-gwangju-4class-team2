"""
============================================================
프로그램명 : report.py
설    명   : 분석 결과를 report.md(마크다운)로 자동 생성하는 모듈.
             수집 → 전처리 → 통계 → 모델까지의 결과를 하나의 문서로 정리한다.
작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.config import DATA_URL, REPORT_PATH, ensure_dirs, get_logger
from src.stats_analysis import format_p

logger = get_logger("report")


def _df_to_md(df: pd.DataFrame) -> str:
    """DataFrame을 마크다운 표 문자열로 변환한다(tabulate 사용)."""
    return df.to_markdown()


def build_report(context: dict[str, Any]) -> str:
    """파이프라인 실행 결과 dict를 받아 마크다운 문자열을 만든다.

    Args:
        context: run_pipeline()이 수집한 단계별 결과.

    Returns:
        완성된 마크다운 문서 문자열.
    """
    load = context["load_compare"]
    eda = context["eda"]
    clean_hist = context["clean_history"]
    ttest = context["ttest"]
    chi2 = context["chi2"]
    metrics = context["metrics"]

    lines: list[str] = []
    add = lines.append

    # ---------- 헤더 ----------
    add("# Adult Census Income 분석 리포트")
    add("")
    add(f"- 생성 일시: {context['generated_at']}")
    add("- 작성: SKALA 광주 캠퍼스 4반 2조 / 박영서")
    add(f"- 데이터 출처: {DATA_URL}")
    add(f"- 총 실행 시간: {context['elapsed_sec']:.2f}초")
    add("")
    add("> 이 문서는 `python -m src.run_pipeline` 실행 시 자동 생성됩니다.")
    add("")

    # ---------- 1. 데이터 준비 ----------
    add("## 1. 데이터 준비 — Pandas vs Polars")
    add("")
    add("| 항목 | Pandas | Polars |")
    add("| --- | --- | --- |")
    add(f"| shape | {load['pandas_shape']} | {load['polars_shape']} |")
    add(f"| 결측치 총계 | {load['pandas_nulls']:,} | {load['polars_nulls']:,} |")
    add(f"| 로딩 시간(초) | {load['pandas_sec']} | {load['polars_sec']} |")
    add("")
    same = "동일" if load["shape_equal"] and load["nulls_equal"] else "불일치"
    speed = load["speedup"]  # Pandas 시간 / Polars 시간
    add(f"- 두 도구의 로딩 결과(행·열 수, 결측치 총계)는 **{same}**합니다.")
    if speed and speed >= 1:
        add(f"- 로딩 속도: Polars(Lazy API)가 Pandas 대비 약 **{speed}배** 빠름.")
    else:
        add(
            f"- 로딩 속도: 이번 실행에서는 Pandas가 약 **{1 / speed:.2f}배** 빨랐습니다. "
            "데이터가 약 3.8MB로 작아 Polars의 멀티스레드·쿼리 최적화 이점보다 "
            "초기화 비용과 공백 제거·결측 변환 등 추가 연산 비용이 더 컸기 때문이며, "
            "수백만 행 이상에서는 Polars가 유리해집니다."
        )
    add("")

    # ---------- 2. EDA ----------
    add("## 2. 기본 EDA")
    add("")
    add(f"- 원본 규모: **{eda['n_rows']:,}행 × {eda['n_cols']}열**")
    add(f"- 중복 행: **{eda['n_duplicates']}행**")
    add("")
    if eda["null_counts"]:
        add("### 결측치 현황")
        add("")
        add("| 컬럼 | 결측 건수 | 비율(%) |")
        add("| --- | ---: | ---: |")
        for col, (cnt, pct) in eda["null_counts"].items():
            add(f"| {col} | {cnt:,} | {pct} |")
        add("")
    else:
        add("- 결측치 없음")
        add("")

    add("### 전처리 이력")
    add("")
    add(f"- 중복 제거: {clean_hist['dropped_duplicates']}행")
    add(f"- 범주형 결측치 최빈값 대체: {clean_hist['filled_categorical']}건")
    add(f"- 수치형 결측치 중앙값 대체: {clean_hist['filled_numeric']}건")
    add(f"- 전처리 후 규모: **{context['clean_rows']:,}행**")
    add("")

    add("### 타깃 분포")
    add("")
    add("| 소득 구간 | 인원 | 비율(%) |")
    add("| --- | ---: | ---: |")
    total = sum(eda["target_dist"].values())
    for label, cnt in eda["target_dist"].items():
        add(f"| {label} | {cnt:,} | {cnt / total * 100:.1f} |")
    add("")

    # ---------- 3. 시각화 ----------
    add("## 3. 시각화")
    add("")
    add(f"- Seaborn 정적 차트(2×2): `{context['static_chart']}`")
    add(f"- Plotly 인터랙티브 차트: `{context['interactive_chart']}`")
    add("")
    add(f"![EDA]({context['static_chart_rel']})")
    add("")

    # ---------- 4. 통계 분석 ----------
    add("## 4. 통계 분석")
    add("")
    add("### 4-1. 기술통계")
    add("")
    add(_df_to_md(context["describe"]))
    add("")

    add("### 4-2. 상관계수")
    add("")
    add(_df_to_md(context["corr"]))
    add("")
    add("상관계수 절댓값 상위 5쌍:")
    add("")
    for col_a, col_b, value in context["top_corr"]:
        add(f"- `{col_a}` ↔ `{col_b}` : **{value:+.3f}**")
    add("")

    add("### 4-3. t-test (독립표본)")
    add("")
    add(f"- 검정: {ttest['test']}")
    add(f"- 비교 변수: `{ttest['value_col']}` / 그룹: `{ttest['group_col']}`")
    add(f"- {ttest['group_a']} 평균 = **{ttest['mean_a']}** (n={ttest['n_a']:,})")
    add(f"- {ttest['group_b']} 평균 = **{ttest['mean_b']}** (n={ttest['n_b']:,})")
    add(
        f"- t 통계량 = **{ttest['t_stat']}**, p-value = **{format_p(ttest['p_value'])}**"
    )
    add(f"- 해석: {ttest['interpretation']}")
    add("")

    add("### 4-4. 카이제곱 독립성 검정")
    add("")
    add(f"- 대상: `{chi2['col_a']}` × `{chi2['col_b']}`")
    add("")
    add(_df_to_md(chi2["contingency"]))
    add("")
    add(
        f"- χ² = **{chi2['chi2']}**, 자유도 = {chi2['dof']}, "
        f"p-value = **{format_p(chi2['p_value'])}**"
    )
    add(f"- 해석: {chi2['interpretation']}")
    add("")

    # ---------- 5. ML Pipeline ----------
    add("## 5. ML Pipeline")
    add("")
    add(f"- 모델: **{metrics['model']}** (sklearn `Pipeline` + `ColumnTransformer`)")
    add(f"- 학습/테스트: {metrics['n_train']:,}행 / {metrics['n_test']:,}행")
    add("")
    add("| 지표 | 값 |")
    add("| --- | ---: |")
    add(f"| Accuracy | {metrics['accuracy']} |")
    add(f"| F1 (macro) | {metrics['f1_macro']} |")
    add(f"| F1 (>50K) | {metrics['f1_positive']} |")
    add(f"| ROC-AUC | {metrics['roc_auc']} |")
    add("")
    add("### 분류 리포트")
    add("")
    add("```")
    add(metrics["classification_report"].rstrip())
    add("```")
    add("")
    add("### 혼동 행렬")
    add("")
    labels = metrics["labels"]
    add(f"| 실제 \\ 예측 | {labels[0]} | {labels[1]} |")
    add("| --- | ---: | ---: |")
    for label, row in zip(labels, metrics["confusion_matrix"], strict=True):
        add(f"| {label} | {row[0]:,} | {row[1]:,} |")
    add("")
    if metrics["feature_importance"]:
        add("### 피처 중요도 상위 10")
        add("")
        add("| 피처 | 중요도 |")
        add("| --- | ---: |")
        for name, score in metrics["feature_importance"]:
            add(f"| {name} | {score} |")
        add("")
    add(f"- 모델 저장 경로: `{context['model_path']}`")
    reload_msg = "성공" if context["reload_ok"] else "실패"
    add(f"- 저장 모델 재로딩 검증: **{reload_msg}**")
    add("")

    # ---------- 6. 결론 ----------
    add("## 6. 결론 및 개선 의견")
    add("")
    add(
        f"- 주당 근무시간은 소득 구간에 따라 평균 차이가 뚜렷했고"
        f"(p={format_p(ttest['p_value'])}), 성별과 소득 구간도 독립이 아니었습니다"
        f"(p={format_p(chi2['p_value'])})."
    )
    add(
        f"- RandomForest 파이프라인은 정확도 {metrics['accuracy']:.1%}, "
        f"ROC-AUC {metrics['roc_auc']:.3f}로 기준선(다수 클래스 예측 약 76%)을 상회했습니다."
    )
    add("- 개선 방향")
    add("  - 클래스 불균형 대응으로 `class_weight` 외에 임계값 튜닝·SMOTE 비교 필요")
    add("  - `capital-gain`의 극단적 편향을 로그 변환 또는 구간화로 완화")
    add("  - 교차검증(`cross_val_score`)과 하이퍼파라미터 탐색으로 일반화 성능 검증")
    add("  - 데이터 규모가 커지면 전처리를 Polars Lazy API로 이관해 처리 시간 단축")
    add("")

    return "\n".join(lines)


def write_report(context: dict[str, Any], path=REPORT_PATH):
    """리포트 마크다운을 생성해 파일로 저장하고 경로를 반환한다."""
    ensure_dirs()
    content = build_report(context)
    path.write_text(content, encoding="utf-8")

    logger.info("리포트 생성 완료: %s (%d줄)", path, content.count("\n") + 1)
    return path


def now_string() -> str:
    """리포트 헤더에 넣을 현재 시각 문자열을 반환한다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
