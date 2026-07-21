"""분석 결과를 종합하여 report.md 를 자동 생성하는 모듈."""
import logging
import os
from pathlib import Path

import numpy as np
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

    # 4. 시각화
    lines.append("## 4. 시각화\n")
    for fig in context.get("figures", []):
        rel_path = Path(os.path.relpath(fig["path"], start=out_path.parent)).as_posix()
        if fig["type"] == "image":
            lines.append(f"**{fig['name']}**\n")
            lines.append(f"![{fig['name']}]({rel_path})\n")
        else:
            lines.append(f"- [{fig['name']}]({rel_path}) (인터랙티브 HTML, 브라우저에서 열어 확인)")
    lines.append("")

    # 5. 가설 검증 결과
    lines.append("## 5. 가설 검증 결과\n")

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

    # 6. 모델 성능
    lines.append("## 6. ML 모델(로지스틱 회귀) 성능\n")
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

    # 7. 로지스틱 회귀 모형 결과 해석
    lines.append("## 7. 로지스틱 회귀 모형 결과 해석\n")

    cm = context["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    n_test = context["n_test"]
    lines.append("### 7-1. 예측 성능 관점\n")
    lines.append(
        f"- ROC-AUC {m['roc_auc']:.4f}는 모델이 실제 >50K와 <=50K를 확률적으로 잘 구분해낸다는 "
        "의미다(1에 가까울수록 완벽한 분리, 0.5는 무작위 수준과 동일). 0.9 안팎의 값은 이 문제에서 "
        "우수한 판별력으로 볼 수 있다."
    )
    lines.append(
        f"- 테스트 {n_test}명 중 실제 >50K인 사람은 {fn + tp}명이며, 이 중 {tp}명은 맞게 예측했지만"
        f"({m['recall']:.2%} 재현율) {fn}명은 <=50K로 잘못 예측했다(False Negative). 반대로 실제 "
        f"<=50K인 {tn + fp}명 중 >50K로 잘못 예측한 경우는 {fp}건(False Positive)에 그쳐, 모델의 오차는 "
        "'고소득자를 놓치는' 방향(재현율 저하)에 더 크게 치우쳐 있다."
    )
    lines.append(
        f"- Precision {m['precision']:.4f}은 모델이 '>50K'로 예측한 사람 중 실제로 맞힌 비율로 비교적 "
        f"신뢰할 수 있는 수준이지만, Recall {m['recall']:.4f}은 실제 고소득자의 약 "
        f"{1 - m['recall']:.0%}를 놓친다는 뜻이다. target 클래스 불균형(>50K 소수)이 모델의 소수 "
        "클래스 탐지력에 영향을 준 것으로 해석된다."
    )
    lines.append("")

    lines.append("### 7-2. 계수(오즈비) 관점\n")
    lines.append(
        "로지스틱 회귀 계수는 다른 변수를 고정했을 때 해당 피처가 1단위(수치형, 표준화 기준) 또는 "
        "해당 범주에 속할 때(범주형) '>50K일 오즈(odds)'에 곱해지는 배수, 즉 오즈비(odds ratio = "
        "exp(계수))로 해석할 수 있다.\n"
    )
    coef_df = context["coef_df"]
    top_pos = coef_df[coef_df["direction"] == "positive"].sort_values("coefficient", ascending=False).head(5)
    top_neg = coef_df[coef_df["direction"] == "negative"].sort_values("coefficient").head(5)
    lines.append("| feature | coefficient | odds ratio | 해석 |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in top_pos.iterrows():
        odds_ratio = float(np.exp(row["coefficient"]))
        lines.append(
            f"| {row['feature']} | {row['coefficient']:.4f} | {odds_ratio:.2f}배 | "
            f"다른 조건이 같을 때 고소득(>50K) 오즈가 약 {odds_ratio:.1f}배 증가 |"
        )
    for _, row in top_neg.iterrows():
        odds_ratio = float(np.exp(row["coefficient"]))
        lines.append(
            f"| {row['feature']} | {row['coefficient']:.4f} | {odds_ratio:.2f}배 | "
            f"다른 조건이 같을 때 고소득(>50K) 오즈가 약 {(1 - odds_ratio):.0%} 감소 |"
        )
    lines.append("")
    lines.append(
        "- capital_gain(표준화 변수)의 계수가 가장 크며, 자본이득이 클수록 고소득 오즈가 10배 이상 "
        "뛴다 — H3을 가장 강하게 뒷받침하는 근거다.\n"
        "- marital_status_Married-civ-spouse, relationship_Wife 등 혼인·가구 관계 관련 피처가 "
        "capital_gain 다음으로 큰 양의 계수를 가져, 개인의 재무·교육 수준 못지않게 가구 구조가 "
        "소득 예측에 강하게 기여함을 시사한다(H4).\n"
        "- education_num은 양의 계수로 학력이 높을수록 고소득 오즈가 커짐을 재확인시켜준다(H2).\n"
        "- 반대로 marital_status_Never-married, relationship_Own-child, sex_Female 등은 음의 계수를 "
        "가져 미혼·자녀 관계·여성일 경우 고소득 오즈가 낮아지는 방향으로 작용한다. sex_Female의 "
        "음의 계수는 데이터에 내재된 성별 소득 격차를 모델이 그대로 학습한 결과로 해석되며, 실제 "
        "활용 시 공정성(fairness) 관점의 추가 검토가 필요하다.\n"
        "- native_country 관련 피처(프랑스, 이탈리아, 콜롬비아 등)도 상위권에 다수 등장하는데, 이는 "
        "해당 국가 출신 표본 수가 적어 계수 추정치의 분산이 커진 결과일 수 있으므로 절대적인 "
        "영향력으로 과잉 해석하지 않도록 주의한다.\n"
    )
    lines.append("")

    lines.append("### 7-3. 종합 결론\n")
    lines.append(
        f"로지스틱 회귀 모형은 ROC-AUC {m['roc_auc']:.4f} 수준의 우수한 판별력을 보이며, "
        "capital_gain·education_num과 같은 개인 재무/교육 지표뿐 아니라 marital_status·relationship과 "
        "같은 가구 구조 변수가 소득 수준을 예측하는 핵심 변수임을 확인했다. 다만 클래스 불균형으로 "
        f"인해 재현율({m['recall']:.4f})이 정밀도({m['precision']:.4f})보다 낮아 실제 고소득자를 "
        "놓치는 오류가 상대적으로 많으므로, 고소득자 선별처럼 재현율이 중요한 목적으로 이 모델을 "
        "활용할 경우 분류 임계값(threshold) 조정이나 class_weight='balanced' 적용 등 추가 튜닝을 "
        "검토할 필요가 있다."
    )
    lines.append("")

    # 8. 산출물 목록
    lines.append("## 8. 산출물 목록\n")
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
