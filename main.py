"""Adult Census Income — 로지스틱 회귀 기반 ML Pipeline 실행 스크립트.

파이프라인 작업 플로우 요약
1. 데이터 수집/로딩: UCI adult.data 다운로드 후 Pandas/Polars로 각각 로딩하여 성능·타입 비교
2. 전처리(clean): 결측치 토큰(' ?') -> NaN 변환 및 dropna, 중복 제거, 문자열 strip, target(income) 0/1 이진화 -> processed CSV 저장
3. 통계 분석(stats): 기술통계·상관분석 + 가설 H1(근무시간), H2(교육수준), H4(혼인/관계) 검정
4. 시각화(visualize): Seaborn 정적 차트(boxplot, heatmap) + Plotly 인터랙티브 차트(education/marital_status) 생성
5. ML 파이프라인(model): train/test 분할 -> 로지스틱 회귀 학습 -> 평가지표 산출 -> 계수 추출
   -> 계수 기반 H3(capital_gain) 검정 -> 모델 저장
6. 리포트 생성(report): 위 모든 결과를 종합해 outputs/report.md 자동 작성
"""
import logging
from pathlib import Path

from src import clean, load, model, report, stats, visualize

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RANDOM_STATE = 42
SEPARATOR = "-" * 60


def run_pipeline() -> None:
    """데이터 수집 -> 전처리 -> 통계분석 -> 시각화 -> ML -> 리포트 생성을 순차 실행한다."""
    logger.info(SEPARATOR)
    logger.info("파이프라인 시작")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 준비: 다운로드 + Pandas/Polars 양쪽 로딩 비교
    logger.info(SEPARATOR)
    logger.info("[1/6] 데이터 수집 및 로딩 시작")
    raw_path = load.fetch_raw_data(RAW_DIR)
    pdf, pd_time = load.load_with_pandas(raw_path)
    pldf, pl_time = load.load_with_polars(raw_path)
    comparison = load.compare_pandas_polars(pdf, pldf, pd_time, pl_time)
    logger.info("[1/6] 데이터 수집 및 로딩 완료")

    # 이후 분석은 Pandas DataFrame 기준으로 진행
    logger.info("기본 EDA (전처리 전): shape=%s", pdf.shape)

    # 2. 전처리
    logger.info(SEPARATOR)
    logger.info("[2/6] 전처리 시작")
    df, clean_summary = clean.clean_data(pdf)
    logger.info("전처리 요약: %s", clean_summary)
    df.to_csv(PROCESSED_DIR / "adult_clean.csv", index=False)
    logger.info("[2/6] 전처리 완료 및 저장")

    # 3. 통계 분석
    logger.info(SEPARATOR)
    logger.info("[3/6] 통계 분석 시작")
    describe_df = stats.descriptive_stats(df)
    corr_df = stats.correlation_analysis(df)
    h1 = stats.test_h1_hours_worked(df)
    h2 = stats.test_h2_education(df)
    h4 = stats.test_h4_marital_relationship(df)
    logger.info("H1 (t-test) 결과: %s", h1["conclusion"])
    logger.info("[3/6] 통계 분석 완료")

    # 4. 시각화 (Seaborn 정적 + Plotly 인터랙티브)
    logger.info(SEPARATOR)
    logger.info("[4/6] 시각화 시작")
    boxplot_path = visualize.plot_income_hours_boxplot(df, FIGURES_DIR / "income_hours_boxplot.png")
    heatmap_path = visualize.plot_correlation_heatmap(df, FIGURES_DIR / "correlation_heatmap.png")
    edu_html_path = visualize.plot_income_ratio_by_education_interactive(
        df, FIGURES_DIR / "income_ratio_by_education.html"
    )
    marital_html_path = visualize.plot_income_ratio_by_marital_status_interactive(
        df, FIGURES_DIR / "income_ratio_by_marital_status.html"
    )
    logger.info("[4/6] 시각화 완료")

    # 5. ML Pipeline (로지스틱 회귀)
    logger.info(SEPARATOR)
    logger.info("[5/6] ML 파이프라인 시작")
    result = model.train_and_evaluate(df)
    logger.info("모델 성능: %s", result["metrics"])
    logger.info("Classification report:\n%s", result["classification_report"])

    capital_gain_coef = model.get_capital_gain_coefficient(result["coef_df"])
    h3 = stats.test_h3_capital_gain(df, model_coef=capital_gain_coef)

    model_path = model.save_model(result["pipeline"], OUTPUTS_DIR / "model.joblib")
    logger.info("[5/6] ML 파이프라인 완료")

    # 6. 리포트 자동 생성
    logger.info(SEPARATOR)
    logger.info("[6/6] 리포트 생성 시작")
    context = {
        "pandas_polars_comparison": comparison,
        "clean_summary": clean_summary,
        "describe_md": report.df_to_markdown(describe_df, index_label="statistic"),
        "corr_md": report.df_to_markdown(corr_df, index_label="feature"),
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"],
        "classification_report": result["classification_report"],
        "coef_df": result["coef_df"],
        "coef_md": report.df_to_markdown(result["coef_df"], show_index=False),
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "artifacts": {
            "전처리 완료 데이터": PROCESSED_DIR / "adult_clean.csv",
            "학습된 모델": model_path,
        },
        "figures": [
            {"name": "Seaborn: income vs hours_per_week boxplot", "path": boxplot_path, "type": "image"},
            {"name": "Seaborn: 상관관계 heatmap", "path": heatmap_path, "type": "image"},
            {"name": "Plotly: education별 고소득 비율", "path": edu_html_path, "type": "html"},
            {"name": "Plotly: marital_status별 고소득 비율", "path": marital_html_path, "type": "html"},
        ],
    }
    report_path = report.generate_report(context, OUTPUTS_DIR / "report.md")
    logger.info("[6/6] 리포트 생성 완료")
    logger.info(SEPARATOR)
    logger.info("파이프라인 종료")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    run_pipeline()
