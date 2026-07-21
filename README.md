# SKALA 광주 캠퍼스 4반 2조

SKALA Gwangju Campus Class 4 Team 2 - Team Project Repository

## 팀원

| 이름 | GitHub |
| --- | --- |
| 박영서 | [@ParkYoungSeo](https://github.com/ParkYoungSeo) |
| 허진녕 | [@Heo-jinnyeong](https://github.com/Heo-jinnyeong) |
| 안성민 | [@rntqkdl](https://github.com/rntqkdl) |
| 황민채 | [@suhrin-huh](https://github.com/suhrin-huh) |
| 고윤진 | [@Yoonjin-Ko](https://github.com/Yoonjin-Ko) |

---

## 프로젝트 개요

**[Day 2] 종합 실습 — End2End 데이터 분석 프로젝트**

Adult Census Income 데이터를 대상으로 수집부터 리포트 자동 생성까지의
전체 분석 파이프라인을 구현했습니다.

```
수집 → Pandas/Polars 비교 로딩 → 결측·중복 처리 → 시각화
     → 통계 검정(t-test, 카이제곱) → sklearn Pipeline 학습·저장 → report.md 자동 생성
```

- **데이터셋**: Adult Census Income (UCI) — 32,561행 × 15열
- **출처**: <https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data>
- **예측 목표**: 연 소득 `>50K` 여부 (이진 분류)

> `data/` 는 `.gitignore` 대상입니다. 위 URL에서 자동으로 내려받으므로 별도 준비가 필요 없습니다.

## 개발 환경 설정

```bash
# 1. 저장소 클론
git clone https://github.com/givpro22/skala-gwangju-4class-team2.git
cd skala-gwangju-4class-team2

# 2. Python 3.11 가상환경 생성 및 활성화
python3.11 -m venv .venv
source .venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt
```

## 실행 방법

```bash
# 전체 파이프라인 실행 (원본 다운로드 → 분석 → 모델 저장 → report.md 생성)
python -m src.run_pipeline

# 원본 데이터를 다시 내려받고 싶을 때
python -m src.run_pipeline --force-download

# 테스트 · 린트
pytest tests/ -v
ruff check .
```

### 실행 산출물

| 경로 | 설명 |
| --- | --- |
| `outputs/report.md` | 분석 결과 자동 생성 리포트 |
| `outputs/figures/eda_seaborn.png` | Seaborn 정적 차트 (2×2 서브플롯) |
| `outputs/figures/eda_plotly.html` | Plotly 인터랙티브 차트 |
| `outputs/models/income_pipeline.joblib` | 학습된 sklearn Pipeline |
| `data/processed/adult_clean.parquet` | 전처리 완료 데이터 |
| `logs/pipeline.log` | 실행 로그 (날짜별 회전) |

## 폴더 구조

```
skala-gwangju-4class-team2/
├── data/
│   ├── raw/                  # 원본 데이터 (수정 금지, git 제외)
│   ├── processed/            # 전처리 완료 데이터
│   └── external/             # 외부 참조 데이터
├── notebooks/
│   └── 01_eda.ipynb          # EDA·실험용 노트북
├── src/                      # 재사용 가능한 Python 모듈
│   ├── __init__.py
│   ├── config.py             # 경로·상수·로거 설정
│   ├── clean.py              # 수집 및 전처리
│   ├── viz.py                # Seaborn / Plotly 시각화
│   ├── stats_analysis.py     # 기술통계·상관·t-test·카이제곱
│   ├── model.py              # sklearn Pipeline 학습·평가·저장
│   ├── report.py             # report.md 자동 생성
│   └── run_pipeline.py       # End2End 실행 스크립트
├── tests/
│   └── test_pipeline.py      # pytest 단위 테스트
├── outputs/                  # 리포트·차트·모델 (git 제외)
├── requirements.txt
└── pyproject.toml            # ruff / pytest 설정
```

## 분석 결과 요약

| 항목 | 결과 |
| --- | --- |
| 결측치 | `workclass` 1,836 / `occupation` 1,843 / `native-country` 583 → 최빈값 대체 |
| 중복 행 | 24행 제거 (32,561 → 32,537) |
| t-test | 소득 그룹 간 주당 근무시간 평균 차이 유의 (38.8h vs 45.5h, p < 1e-308) |
| 카이제곱 | 성별 × 소득 구간 독립 아님 (χ²=1516.5, dof=1, p < 1e-308) |
| 모델 | RandomForest — Accuracy 0.818 / F1(macro) 0.784 / ROC-AUC 0.921 |

자세한 내용은 실행 후 생성되는 [`outputs/report.md`](outputs/report.md)를 참고하세요.

---

## 협업 규칙

### 브랜치 전략

| 브랜치 | 용도 |
| --- | --- |
| `main` | 배포 가능한 안정 버전 |
| `develop` | 통합 개발 브랜치 |
| `feature/*` | 기능 단위 개발 |
| `fix/*` | 버그 수정 |

### 커밋 컨벤션

```
<type>: <subject>
```

| Type | 설명 |
| --- | --- |
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `style` | 코드 포맷팅 (기능 변경 없음) |
| `refactor` | 리팩토링 |
| `test` | 테스트 코드 추가/수정 |
| `chore` | 빌드, 설정 등 기타 작업 |
