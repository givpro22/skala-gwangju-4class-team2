"""
============================================================
프로그램명 : run_pipeline.py
설    명   : [Day 2] 종합 실습 - End2End 데이터 분석 파이프라인 실행 스크립트.
             수집 → 검증/전처리 → 시각화 → 통계 분석 → ML Pipeline → 리포트 생성
             까지의 전 과정을 한 번의 실행으로 수행한다.

실행 방법  : python -m src.run_pipeline
             python -m src.run_pipeline --force-download   (원본 재다운로드)

작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from src import clean, model, report, stats_analysis, viz
from src.config import PROJECT_ROOT, ensure_dirs, get_logger

logger = get_logger("run_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Adult Census Income End2End 분석 파이프라인"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="로컬 캐시가 있어도 원본 데이터를 다시 내려받는다",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="테스트셋 비율 (기본값: 0.2)",
    )
    return parser.parse_args(argv)


def run(force_download: bool = False, test_size: float = 0.2) -> dict[str, Any]:
    """전체 파이프라인을 순서대로 실행하고 단계별 결과를 dict로 반환한다.

    각 단계를 독립 함수로 분리해 두었기 때문에
    실패 시 어느 단계에서 멈췄는지 로그로 즉시 확인할 수 있다.

    Args:
        force_download: 원본 데이터 강제 재다운로드 여부.
        test_size: 학습/테스트 분할 비율.

    Returns:
        리포트 생성에 사용되는 컨텍스트 dict.
    """
    ensure_dirs()
    started = time.perf_counter()
    context: dict[str, Any] = {}

    # --- [1] Extract : 원본 수집 -----------------------------------
    logger.info("=" * 60)
    logger.info("[1/6] 데이터 수집")
    raw_path = clean.download_raw(force=force_download)

    # --- [2] Load & Compare : Pandas vs Polars ---------------------
    logger.info("[2/6] Pandas / Polars 로딩 및 비교")
    pdf, _ = clean.load_with_pandas(raw_path)
    pldf, _ = clean.load_with_polars(raw_path)

    # 로딩 시간은 1회 측정이 불안정하므로(첫 호출에 초기화 비용이 섞임)
    # 예열 후 반복 측정한 중앙값을 비교에 사용한다.
    t_pandas, t_polars = clean.benchmark_loaders(raw_path)
    context["load_compare"] = clean.compare_loaders(pdf, pldf, t_pandas, t_polars)

    # --- [3] Transform : EDA + 결측/중복 처리 -----------------------
    logger.info("[3/6] 기본 EDA 및 전처리")
    context["eda"] = clean.basic_eda(pdf)
    context["cat_profile"] = clean.categorical_profile(pdf)
    df_clean, history = clean.clean_data(pdf)
    context["clean_history"] = history
    context["clean_rows"] = len(df_clean)
    # 최빈값 대체가 분포를 얼마나 바꿨는지 기록 (대체 전략의 부작용 확인용)
    context["impute_impact"] = clean.imputation_impact(pdf, df_clean)
    context["processed_path"] = str(clean.save_processed(df_clean))

    # --- [4] Visualize : Seaborn + Plotly --------------------------
    logger.info("[4/6] 시각화")
    static_path = viz.plot_static_eda(df_clean)
    interactive_path = viz.plot_interactive(df_clean)
    context["static_chart"] = str(static_path.relative_to(PROJECT_ROOT))
    context["interactive_chart"] = str(interactive_path.relative_to(PROJECT_ROOT))
    # 리포트(outputs/report.md) 기준 상대 경로로 이미지 링크를 건다
    context["static_chart_rel"] = str(
        static_path.relative_to(static_path.parent.parent)
    )

    # --- [5] Statistics : 기술통계 · 상관 · 검정 --------------------
    logger.info("[5/6] 통계 분석")
    context["describe"] = stats_analysis.describe_numeric(df_clean)
    corr = stats_analysis.correlation_matrix(df_clean)
    context["corr"] = corr
    context["top_corr"] = stats_analysis.top_correlations(corr)
    context["ttest"] = stats_analysis.run_ttest(df_clean)
    context["chi2"] = stats_analysis.run_chi2(df_clean)

    # --- [6] ML Pipeline : 학습 · 평가 · 저장 -----------------------
    logger.info("[6/6] ML Pipeline 학습 및 저장")
    pipeline, metrics = model.train_and_evaluate(df_clean, test_size=test_size)
    context["metrics"] = metrics
    model_path = model.save_model(pipeline)
    context["model_path"] = str(model_path.relative_to(PROJECT_ROOT))

    # 저장한 모델을 다시 읽어 동일하게 예측되는지 확인 (배포 재현성 검증)
    sample = df_clean.head(5)[model.NUMERIC_FEATURES + model.CATEGORICAL_FEATURES]
    context["reload_ok"] = model.verify_reload(model_path, sample)

    # --- 리포트 자동 생성 -------------------------------------------
    context["elapsed_sec"] = time.perf_counter() - started
    report_path = report.write_report(context)
    context["report_path"] = str(report_path.relative_to(PROJECT_ROOT))

    logger.info("=" * 60)
    logger.info("파이프라인 완료 (%.2f초) → %s", context["elapsed_sec"], report_path)
    return context


def main(argv: list[str] | None = None) -> int:
    """스크립트 진입점. 예외를 잡아 종료 코드로 변환한다.

    Returns:
        정상 종료 시 0, 실패 시 1.
    """
    args = parse_args(argv)
    try:
        run(force_download=args.force_download, test_size=args.test_size)
    except clean.DataLoadError as exc:
        logger.error("데이터 수집 단계 실패: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("입력/설정 오류: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - 최상위에서 모든 예외를 기록
        logger.exception("예기치 못한 오류로 파이프라인이 중단되었습니다: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
