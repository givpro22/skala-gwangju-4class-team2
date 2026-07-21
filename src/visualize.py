"""EDA 시각화 모듈 (Seaborn 정적 차트, Plotly 인터랙티브 차트)."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 파일 저장이 가능하도록 설정
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.stats import NUMERIC_COLS

logger = logging.getLogger(__name__)

# 카테고리(정체성) 색상: 고정 순서의 파랑/주황 슬롯 (2개 그룹 비교용)
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834"]
# 상관계수(-1~1, 0이 중립)를 위한 diverging 색상: 파랑<->빨강 + 회색 중간값
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
)


def plot_income_hours_boxplot(df: pd.DataFrame, out_path: Path) -> Path:
    """[Seaborn] income 그룹별 hours_per_week 분포 boxplot을 그려 png로 저장한다 (H1 시각화)."""
    logger.info("income vs hours_per_week boxplot 생성 시작")
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_df = df.assign(income_label=df["income"].map({0: "<=50K", 1: ">50K"}))
    sns.boxplot(
        data=plot_df, x="income_label", y="hours_per_week", hue="income_label",
        palette=CATEGORICAL_COLORS, legend=False, ax=ax,
    )
    ax.set_title("Income group vs Hours per week")
    ax.set_xlabel("Income")
    ax.set_ylabel("Hours per week")
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("boxplot 저장 완료")
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> Path:
    """[Seaborn] 수치형 변수 상관관계 heatmap을 그려 png로 저장한다."""
    logger.info("상관관계 heatmap 생성 시작")
    corr = df[NUMERIC_COLS + ["income"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0, ax=ax)
    ax.set_title("Correlation heatmap (numeric features + income)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("heatmap 저장 완료")
    return out_path


def plot_income_ratio_by_education_interactive(df: pd.DataFrame, out_path: Path) -> Path:
    """[Plotly] education별 고소득(>50K) 비율 인터랙티브 막대 차트를 html로 저장한다 (H2 시각화)."""
    logger.info("education별 고소득 비율 차트 생성 시작")
    ratio = (
        df.groupby("education")["income"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"income": "high_income_ratio"})
    )
    fig = px.bar(
        ratio,
        x="education",
        y="high_income_ratio",
        title="Education level vs High-income(>50K) ratio",
        labels={"education": "Education", "high_income_ratio": "High income ratio"},
    )
    fig.update_traces(marker_color=CATEGORICAL_COLORS[0])
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(out_path)
    logger.info("education별 고소득 비율 차트 저장 완료")
    return out_path


def plot_income_ratio_by_marital_status_interactive(df: pd.DataFrame, out_path: Path) -> Path:
    """[Plotly] marital_status별 고소득(>50K) 비율 인터랙티브 막대 차트를 html로 저장한다 (H4 시각화)."""
    logger.info("marital_status별 고소득 비율 차트 생성 시작")
    ratio = (
        df.groupby("marital_status")["income"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"income": "high_income_ratio"})
    )
    fig = px.bar(
        ratio,
        x="marital_status",
        y="high_income_ratio",
        title="Marital status vs High-income(>50K) ratio",
        labels={"marital_status": "Marital status", "high_income_ratio": "High income ratio"},
    )
    fig.update_traces(marker_color=CATEGORICAL_COLORS[1])
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(out_path)
    logger.info("marital_status별 고소득 비율 차트 저장 완료")
    return out_path
