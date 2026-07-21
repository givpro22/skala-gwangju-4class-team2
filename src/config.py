"""
============================================================
프로그램명 : config.py
설    명   : Day2 종합 실습 프로젝트의 공통 설정(경로/상수/로거)을 정의한다.
             모든 모듈이 이 파일의 경로 상수를 참조하므로,
             폴더 구조가 바뀌어도 이 파일 한 곳만 수정하면 된다.
작 성 자   : 박영서 (SKALA 광주 4반 2조)
변경 내역
  2026-07-21  박영서  최초 작성
============================================================
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# ------------------------------------------------------------------
# 1. 경로 상수 — 프로젝트 루트 기준 절대 경로로 관리한다.
#    (스크립트를 어느 위치에서 실행해도 동일하게 동작하도록 함)
# ------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"  # 원본 데이터 (수정 금지)
PROCESSED_DIR: Path = DATA_DIR / "processed"  # 전처리 완료 데이터
EXTERNAL_DIR: Path = DATA_DIR / "external"  # 외부 참조 데이터

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"  # 차트(png/html)
MODEL_DIR: Path = OUTPUT_DIR / "models"  # 학습된 파이프라인(joblib)
LOG_DIR: Path = PROJECT_ROOT / "logs"

REPORT_PATH: Path = OUTPUT_DIR / "report.md"  # 자동 생성 리포트

# ------------------------------------------------------------------
# 2. 데이터셋 정보 — Adult Census Income (UCI)
#    강의자료 [Day 2] 종합 실습에서 제시된 데이터셋 중 하나를 선택
# ------------------------------------------------------------------
DATA_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
RAW_FILE: Path = RAW_DIR / "adult.data"
PROCESSED_FILE: Path = PROCESSED_DIR / "adult_clean.parquet"

COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

TARGET: str = "income"  # 예측 대상(이진 분류: <=50K / >50K)

# 모델 학습에 사용할 컬럼
# fnlwgt(표본 가중치)는 개인의 소득과 직접 관련이 없어 제외한다.
# education-num은 education과 1:1 대응(중복 정보)이므로 수치형만 남긴다.
NUMERIC_FEATURES: list[str] = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

RANDOM_STATE: int = 42  # 재현성 확보용 시드


def ensure_dirs() -> None:
    """프로젝트에서 사용하는 산출물 폴더를 미리 생성한다.

    폴더가 이미 존재해도 예외가 발생하지 않도록 exist_ok=True 로 처리한다.
    """
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        FIGURE_DIR,
        MODEL_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "pipeline") -> logging.Logger:
    """콘솔(INFO 이상) + 파일(DEBUG 이상)로 동시에 기록하는 로거를 반환한다.

    Args:
        name: 로거 이름. 모듈별로 다른 이름을 주면 로그에서 출처를 구분할 수 있다.

    Returns:
        핸들러가 부착된 logging.Logger 인스턴스.

    Note:
        같은 이름으로 여러 번 호출해도 핸들러가 중복 부착되지 않도록
        기존 핸들러 유무를 먼저 확인한다.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # 이미 설정된 로거면 그대로 재사용
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    # 콘솔 핸들러 — 실행 중 진행 상황 확인용
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 파일 핸들러 — 날짜별 회전, 최근 7일 보관
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        LOG_DIR / "pipeline.log",
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
