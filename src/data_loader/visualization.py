import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


logger = logging.getLogger(__name__)


def create_seaborn_charts(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """분포와 상관관계를 보여주는 Seaborn 차트 2개를 저장합니다."""
    sns.set_theme(style="whitegrid")
    saved_files = []

    distribution_path = output_dir / "seaborn_age_distribution.png"
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=dataframe,
        x="age",
        hue="income",
        bins=30,
        kde=True,
        multiple="layer",
        ax=axis,
    )
    axis.set_title("Age Distribution by Income")
    axis.set_xlabel("Age")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(distribution_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    saved_files.append(distribution_path)

    correlation_path = output_dir / "seaborn_correlation_heatmap.png"
    numeric_dataframe = dataframe.select_dtypes(include="number")
    figure, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        numeric_dataframe.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=axis,
    )
    axis.set_title("Numeric Feature Correlation")
    figure.tight_layout()
    figure.savefig(correlation_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    saved_files.append(correlation_path)

    return saved_files


def create_plotly_charts(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """그룹 차이를 보여주는 Plotly 인터랙티브 차트 2개를 저장합니다."""
    saved_files = []

    box_path = output_dir / "plotly_work_hours_by_income.html"
    box_figure = px.box(
        dataframe,
        x="income",
        y="hours-per-week",
        color="income",
        points="outliers",
        title="Weekly Work Hours by Income",
        labels={
            "income": "Income",
            "hours-per-week": "Hours per Week",
        },
    )
    box_figure.write_html(box_path, include_plotlyjs=True)
    saved_files.append(box_path)

    group_counts = (
        dataframe.groupby(["education", "income"], observed=True)
        .size()
        .reset_index(name="count")
    )
    bar_path = output_dir / "plotly_education_income_comparison.html"
    bar_figure = px.bar(
        group_counts,
        x="education",
        y="count",
        color="income",
        barmode="group",
        title="Income Groups by Education",
        labels={
            "education": "Education",
            "income": "Income",
            "count": "Count",
        },
    )
    bar_figure.update_layout(xaxis_tickangle=-45)
    bar_figure.write_html(bar_path, include_plotlyjs=True)
    saved_files.append(bar_path)

    return saved_files


def create_visualizations(
    dataframe: pd.DataFrame,
    output_directory: str = "outputs/charts",
) -> Tuple[List[Path], List[Path]]:
    """Seaborn 차트 2개와 Plotly 차트 2개를 파일로 생성합니다."""
    if dataframe.empty:
        raise ValueError("시각화할 데이터가 없습니다.")

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    seaborn_files = create_seaborn_charts(dataframe, output_dir)
    plotly_files = create_plotly_charts(dataframe, output_dir)

    logger.info(
        "시각화 생성 완료: Seaborn=%s개, Plotly=%s개, 경로=%s",
        len(seaborn_files),
        len(plotly_files),
        output_dir.resolve(),
    )
    return seaborn_files, plotly_files
