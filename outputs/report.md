# Adult Census Income — 로지스틱 회귀 분석 리포트

## 1. 데이터 개요 및 Pandas vs Polars 로딩 비교

- 원본 데이터 shape: `(32561, 15)`
- Pandas 로딩 시간: `0.0196초`
- Polars 로딩 시간: `0.0199초`

| 컬럼 | Pandas dtype | Polars dtype |
| --- | --- | --- |
| age | int64 | Int64 |
| workclass | object | String |
| fnlwgt | int64 | String |
| education | object | String |
| education_num | int64 | String |
| marital_status | object | String |
| occupation | object | String |
| relationship | object | String |
| race | object | String |
| sex | object | String |
| capital_gain | int64 | String |
| capital_loss | int64 | String |
| hours_per_week | int64 | String |
| native_country | object | String |
| income | object | String |

**주의할 차이점**

- 원본 파일 끝에 빈 줄이 하나 있다. Pandas는 기본값(skip_blank_lines=True)으로 이를 건너뛰지만, Polars는 이를 전 컬럼이 null인 행으로 읽어들여 행 수가 1 더 많다.
- ['fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] 컬럼은 CSV 값 앞에 공백(', 77516' 등)이 있어 Polars의 기본 스키마 추론에서는 숫자로 인식되지 못하고 String으로 로딩된다. 반면 Pandas의 C 파서는 숫자 파싱 시 앞뒤 공백을 자동으로 무시해 int64로 인식한다.

## 2. 전처리 요약

- 결측치 토큰(' ?') → NaN 변환 후 결측 비율 계산
- 결측치가 있는 행: 2399/32561 (비율 7.37%) → 결측 비율이 낮아 해당 행을 dropna로 제거
- dropna 이후 행 수: 30162
- 중복 행 제거: 30162 → 30139 (23건 제거)
- 최종 shape: (30139, 15)
- target(income) 클래스 비율: >50K = 24.90% (불균형 데이터 — <=50K 클래스가 다수)

## 3. EDA 요약

### 3-1. 기술통계 (수치형 변수)

| statistic | age | fnlwgt | education_num | capital_gain | capital_loss | hours_per_week |
| --- | --- | --- | --- | --- | --- | --- |
| count | 30139.0 | 30139.0 | 30139.0 | 30139.0 | 30139.0 | 30139.0 |
| mean | 38.4417 | 189795.026 | 10.1225 | 1092.8412 | 88.4399 | 40.9347 |
| std | 13.1314 | 105658.6243 | 2.5487 | 7409.1106 | 404.4452 | 11.9788 |
| min | 17.0 | 13769.0 | 1.0 | 0.0 | 0.0 | 1.0 |
| 25% | 28.0 | 117627.5 | 9.0 | 0.0 | 0.0 | 40.0 |
| 50% | 37.0 | 178417.0 | 10.0 | 0.0 | 0.0 | 40.0 |
| 75% | 47.0 | 237604.5 | 13.0 | 0.0 | 0.0 | 45.0 |
| max | 90.0 | 1484705.0 | 16.0 | 99999.0 | 4356.0 | 99.0 |

### 3-2. 상관계수 (수치형 변수 + income)

| feature | age | fnlwgt | education_num | capital_gain | capital_loss | hours_per_week | income |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age | 1.0 | -0.0763 | 0.0432 | 0.0802 | 0.0601 | 0.1013 | 0.242 |
| fnlwgt | -0.0763 | 1.0 | -0.0452 | 0.0004 | -0.0098 | -0.023 | -0.009 |
| education_num | 0.0432 | -0.0452 | 1.0 | 0.1245 | 0.0796 | 0.1528 | 0.3354 |
| capital_gain | 0.0802 | 0.0004 | 0.1245 | 1.0 | -0.0323 | 0.0804 | 0.2212 |
| capital_loss | 0.0601 | -0.0098 | 0.0796 | -0.0323 | 1.0 | 0.0524 | 0.15 |
| hours_per_week | 0.1013 | -0.023 | 0.1528 | 0.0804 | 0.0524 | 1.0 | 0.2294 |
| income | 0.242 | -0.009 | 0.3354 | 0.2212 | 0.15 | 0.2294 | 1.0 |

## 4. 가설 검증 결과

### H1. >50K 그룹의 hours_per_week 가 유의하게 더 길다 (Welch's t-test)

- <=50K 그룹 평균: 39.35시간
- >50K 그룹 평균: 45.71시간
- t-statistic: 43.1697, p-value: 0
- 해석 (유의수준 0.05): p-value(0) < 유의수준(0.05) 이므로, >50K 그룹과 <=50K 그룹의 평균 hours_per_week 차이는 통계적으로 유의하다. (H1 채택: >50K 그룹 평균 45.71시간 vs <=50K 그룹 평균 39.35시간)

### H2. education_num이 높을수록 고소득 비율이 높다

- education_num과 income의 상관계수: 0.3354
- 해석: education_num과 income의 상관계수는 0.3354로 양의 상관관계를 보이며, education_num이 높은 그룹일수록 고소득(>50K) 비율이 대체로 증가한다. (H2 지지)

### H3. capital_gain은 소득과 강한 양의 관계를 가지는 핵심 예측 변수다

- capital_gain과 income의 상관계수: 0.2212
- 로지스틱 회귀 계수: 2.3753
- 해석: capital_gain과 income의 상관계수는 0.2212로 양의 상관관계를 보인다. 로지스틱 회귀 계수 또한 2.3753로 양의 값을 가져, capital_gain이 고소득을 예측하는 핵심 변수임을 뒷받침한다. (H3 지지)

### H4. marital_status/relationship에 따라 고소득 비율 차이가 크다

- marital_status 그룹 간 최대-최소 격차: 42.78%
- relationship 그룹 간 최대-최소 격차: 47.93%
- 해석: marital_status 그룹 간 고소득 비율 격차는 42.78%, relationship 그룹 간 격차는 47.93%로 모두 큰 차이를 보인다. (H4 지지)

## 5. ML 모델(로지스틱 회귀) 성능

- 학습 데이터: 24111건 / 테스트 데이터: 6028건

| 지표 | 값 |
| --- | --- |
| Accuracy | 0.8470 |
| Precision | 0.7337 |
| Recall | 0.6056 |
| F1-score | 0.6635 |
| ROC-AUC | 0.9005 |

> 주의: target 클래스가 불균형(>50K가 소수)하므로 accuracy만으로 모델 성능을 판단하지 않는다. Precision/Recall/F1/ROC-AUC를 함께 확인해야 소수 클래스 탐지력을 올바르게 평가할 수 있다.

### Confusion Matrix

```
[[4197  330]
 [ 592  909]]
```

### Classification Report

```
              precision    recall  f1-score   support

       <=50K       0.88      0.93      0.90      4527
        >50K       0.73      0.61      0.66      1501

    accuracy                           0.85      6028
   macro avg       0.81      0.77      0.78      6028
weighted avg       0.84      0.85      0.84      6028

```

### 로지스틱 회귀 계수 (상위 양/음 피처) 및 가설 연결 해석

| direction | feature | coefficient |
| --- | --- | --- |
| positive | capital_gain | 2.3753 |
| positive | marital_status_Married-civ-spouse | 1.2847 |
| positive | relationship_Wife | 1.1526 |
| positive | marital_status_Married-AF-spouse | 1.0899 |
| positive | native_country_France | 0.8824 |
| positive | native_country_Italy | 0.8307 |
| positive | occupation_Exec-managerial | 0.8065 |
| positive | native_country_Yugoslavia | 0.7812 |
| positive | education_num | 0.7288 |
| positive | native_country_Cambodia | 0.6932 |
| negative | native_country_Columbia | -1.2747 |
| negative | occupation_Priv-house-serv | -1.2642 |
| negative | relationship_Own-child | -1.1764 |
| negative | native_country_South | -1.1432 |
| negative | occupation_Farming-fishing | -1.105 |
| negative | marital_status_Never-married | -1.0581 |
| negative | sex_Female | -1.0037 |
| negative | native_country_Dominican-Republic | -0.8482 |
| negative | relationship_Other-relative | -0.8437 |
| negative | native_country_Vietnam | -0.8154 |

- capital_gain 계수가 강한 양수로 나타나 **H3**을 뒷받침한다.
- education_num 계수가 양수로 나타나 **H2**를 뒷받침한다.
- marital_status/relationship 관련 원-핫 피처들이 상위 계수에 다수 포함되어 **H4**를 뒷받침한다.

## 6. 산출물 목록

- 전처리 완료 데이터: `/Users/minchae/workspace/skala-gwangju-4class-team2/data/processed/adult_clean.csv`
- 학습된 모델: `/Users/minchae/workspace/skala-gwangju-4class-team2/outputs/model.joblib`
- Seaborn: income vs hours_per_week boxplot: `/Users/minchae/workspace/skala-gwangju-4class-team2/outputs/figures/income_hours_boxplot.png`
- Seaborn: 상관관계 heatmap: `/Users/minchae/workspace/skala-gwangju-4class-team2/outputs/figures/correlation_heatmap.png`
- Plotly: education별 고소득 비율: `/Users/minchae/workspace/skala-gwangju-4class-team2/outputs/figures/income_ratio_by_education.html`
- Plotly: marital_status별 고소득 비율: `/Users/minchae/workspace/skala-gwangju-4class-team2/outputs/figures/income_ratio_by_marital_status.html`
