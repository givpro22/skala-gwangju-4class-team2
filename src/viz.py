"""
============================================================
프로그램명 : viz.py
설    명   : 시각화 담당 모듈.
             - Seaborn 정적 차트(2x2 서브플롯) → PNG 저장
             - Plotly Express 인터랙티브 차트 → HTML 저장
작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # GUI 없는 환경에서도 저장 가능하도록 백엔드 고정

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.config import FIGURE_DIR, TARGET, ensure_dirs, get_logger

logger = get_logger("viz")


def _setup_style() -> None:
    """차트 공통 스타일과 한글 폰트를 설정한다.

    macOS 기본 한글 폰트(AppleGothic)를 사용하며,
    마이너스 기호가 깨지지 않도록 unicode_minus 옵션을 끈다.
    """
    sns.set_theme(style="whitegrid")
    for font in ("AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"):
        if font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_static_eda(df: pd.DataFrame, filename: str = "eda_seaborn.png"):
    """Seaborn 정적 차트 4종을 2x2 서브플롯 한 장으로 그려 PNG로 저장한다.

    구성
      (0,0) 나이 분포 히스토그램 + KDE      → 분포 확인
      (0,1) 소득 그룹별 주당 근무시간 박스플롯 → 그룹 비교
      (1,0) 학력연수 상위 교육수준별 소득 비율 막대 → 범주 비교
      (1,1) 수치형 변수 상관 히트맵          → 상관관계

    Args:
        df: 전처리 완료 DataFrame.
        filename: outputs/figures 아래에 저장할 파일명.

    Returns:
        저장된 PNG 파일의 Path.
    """
    ensure_dirs()
    _setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adult Census Income - EDA (Seaborn)", fontsize=15, fontweight="bold")

    # (0,0) 나이 분포 — 소득 그룹별로 색을 나눠 분포 차이를 함께 확인
    sns.histplot(
        data=df, x="age", hue=TARGET, kde=True, bins=30, ax=axes[0, 0], alpha=0.55
    )
    axes[0, 0].set_title("나이 분포 (소득 그룹별)")
    axes[0, 0].set_xlabel("나이(세)")
    axes[0, 0].set_ylabel("빈도")

    # (0,1) 주당 근무시간 박스플롯 — t-test 대상 변수의 그룹 간 분포 비교
    sns.boxplot(data=df, x=TARGET, y="hours-per-week", hue=TARGET, ax=axes[0, 1])
    axes[0, 1].set_title("소득 그룹별 주당 근무시간")
    axes[0, 1].set_xlabel("소득 구간")
    axes[0, 1].set_ylabel("주당 근무시간")
    if axes[0, 1].get_legend() is not None:
        axes[0, 1].get_legend().remove()  # x축과 중복되는 범례 제거

    # (1,0) 성별 × 소득 비율 — 카이제곱 검정 대상 변수 시각화
    ratio = pd.crosstab(df["sex"], df[TARGET], normalize="index").mul(100).reset_index()
    ratio_long = ratio.melt(id_vars="sex", var_name=TARGET, value_name="ratio")
    sns.barplot(data=ratio_long, x="sex", y="ratio", hue=TARGET, ax=axes[1, 0])
    axes[1, 0].set_title("성별 소득 구간 비율")
    axes[1, 0].set_xlabel("성별")
    axes[1, 0].set_ylabel("비율(%)")

    # (1,1) 수치형 상관 히트맵
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("수치형 변수 상관행렬")

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # 메모리 누수 방지

    logger.info("Seaborn 정적 차트 저장: %s", out_path)
    return out_path


def plot_interactive(df: pd.DataFrame, filename: str = "eda_plotly.html"):
    """Plotly Express 인터랙티브 차트를 만들어 HTML 파일로 저장한다.

    직업(occupation) × 소득 구간별 인원을 그룹 막대로 표현하고,
    호버 시 평균 근무시간·평균 나이를 함께 보여준다.

    Args:
        df: 전처리 완료 DataFrame.
        filename: outputs/figures 아래에 저장할 파일명.

    Returns:
        저장된 HTML 파일의 Path.
    """
    ensure_dirs()

    agg = (
        df.groupby(["occupation", TARGET], observed=True)
        .agg(
            count=("age", "size"),
            avg_hours=("hours-per-week", "mean"),
            avg_age=("age", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    agg[["avg_hours", "avg_age"]] = agg[["avg_hours", "avg_age"]].round(1)

    fig = px.bar(
        agg,
        x="occupation",
        y="count",
        color=TARGET,
        barmode="group",
        hover_data=["avg_hours", "avg_age"],
        labels={
            "occupation": "직업군",
            "count": "인원 수",
            TARGET: "소득 구간",
            "avg_hours": "평균 주당 근무시간",
            "avg_age": "평균 나이",
        },
        title="직업군별 소득 구간 분포 (인터랙티브)",
    )
    fig.update_layout(xaxis_tickangle=-40, height=600, legend_title_text="소득 구간")

    out_path = FIGURE_DIR / filename
    fig.write_html(out_path, include_plotlyjs="cdn")  # 파일 용량 절감

    logger.info("Plotly 인터랙티브 차트 저장: %s", out_path)
    return out_path
