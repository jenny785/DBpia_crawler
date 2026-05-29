# DSBA Lab 머신러닝 프로젝트 분석 보고서

## 📋 개요
DSBA Lab의 주요 머신러닝 프로젝트들을 분석하고, DBpia_crawler와의 통합 방안을 제시합니다.

---

## 1️⃣ TS-Unity: 통합 시계열 분석 프레임워크

### 프로젝트 소개
- **용도**: 시계열 데이터의 예측, 이상 탐지, 결측값 대체, 분류
- **특징**: 다중 모델 지원, 타입 힌트, REST API 제공, 실시간/배치 처리

### 핵심 구조
```
TS-Unity/
├── src/
│   ├── main.py                 # 엔트리포인트
│   ├── config/                 # 설정 관리
│   ├── core/                   # 핵심 파이프라인
│   ├── exp/                    # 실험 모듈
│   │   ├── exp_long_term_forecasting.py     # 장기 예측
│   │   ├── exp_anomaly_detection.py         # 이상 탐지
│   │   ├── exp_imputation.py               # 결측값 처리
│   │   └── exp_classification.py           # 분류
│   └── api/
│       └── inference_server.py # REST API
├── examples/
│   ├── batch_inference_example.py
│   └── realtime_inference_example.py
└── scripts/
```

### 지원 모델
- Autoformer
- Transformer
- Informer
- 기타 시계열 모델들

### 활용 사례
**DBpia 크롤러와의 연동 시나리오**:
```
DBpia 논문 데이터 → 시계열 분석
- 시간대별 논문 발행량 예측
- 특정 주제의 논문 이상 탐지
- 연구 트렌드 분류
```

### 코드 특징
✅ 전문적인 로깅 시스템
✅ 의존성 검증
✅ 모듈식 구조
✅ 타입 힌팅

---

## 2️⃣ RAPID: 학습 없는 로그 이상 탐지

### 프로젝트 소개
- **제목**: Training-free Retrieval-based Log Anomaly Detection with PLM
- **용도**: 시스템 로그의 이상 탐지
- **특징**: Pre-trained Language Model 활용, 학습 단계 불필요

### 핵심 파이프라인
```
1. split_data.py
   - 학습/테스트 데이터 분할
   - 데이터 전처리

2. preprocess_rep.py
   - BERT/AutoModel을 이용한 텍스트 표현 생성
   - 다중 처리(multiprocessing) 활용

3. ad_test_coreSet.py
   - 이상 탐지 알고리즘 실행
   - Anomaly Score 계산

4. utils.py
   - 정규표현식 기반 로그 파싱 (BGL, Thunder Bird, HDFS)
```

### 핵심 기술
```python
# BERT 기반 표현 생성
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

# 배치 처리로 메모리 효율화
representations = model(**tokens).last_hidden_state

# 멀티프로세싱을 통한 병렬 처리
parmap.map(preprocess, raw_data, pm_processes=num_processors-2)
```

### DBpia 적용 가능성
```
다양한 로그 데이터 소스:
- 크롤러 실행 로그 분석
- 네트워크 요청 이상 탐지
- 데이터 처리 파이프라인 모니터링
```

### 주요 장점
✅ 학습 단계 제거 → 빠른 배포
✅ PLM 활용 → 높은 성능
✅ 멀티프로세싱 → 대용량 데이터 처리
✅ 정규표현식 기반 유연한 파싱

---

## 3️⃣ Fake-News-Detection: 한국어 가짜뉴스 탐지

### 프로젝트 소개
- **언어**: 한국어 (한국어 처리 특화)
- **데이터셋**: Part1 (제목-본문 일치성), Part2 (본문 기반 탐지)
- **모델**: KoBERT, Transformer 기반

### 데이터 구조
```
Part 1: 제목 - 본문 일치성
├── Clickbait_Auto (자동 생성 클릭베이트)
├── Clickbait_Direct (직접 작성 클릭베이트)
└── NonClickbait_Auto (비클릭베이트)

Part 2: 본문 기반 가짜뉴스 탐지
```

### 기술 스택
```
- PyTorch 1.8+
- Transformers (KoBERT)
- Konlpy (한국어 형태소 분석)
- Weights & Biases (실험 추적)
```

### DBpia 뉴스 레퍼런스 활용
```
논문에서 참고된 뉴스 기사:
→ 진위 여부 자동 검증
→ 데이터 품질 향상
```

---

## 4️⃣ Text-Analytics: 텍스트 분석 강의

### 학습 자료 구성
- Unstructured Data Analysis (대학원 강의)
- 실습 자료 및 프로젝트 가이드
- 용어/개념 정리

### 주요 내용
- NLP 기초 개념
- 텍스트 전처리
- 임베딩 기법
- 분류 모델
- 감정 분석

---

## 💡 DBpia_crawler 통합 전략

### 1. 즉시 적용 가능
```
✅ RAPID를 이용한 크롤러 로그 모니터링
✅ Fake-News-Detection 기술로 논문 신뢰도 검증
✅ TS-Unity로 수집 데이터 시계열 분석
```

### 2. 단기 개선 (1-2주)
```
📌 프로젝트별 모듈화
  - ts_unity/: 시계열 분석
  - rapid/: 로그 모니터링
  - news_detection/: 신뢰도 검증

📌 의존성 통합
  - requirements.txt 통합
  - 가상 환경 설정

📌 API 통합
  - FastAPI 백엔드 구성
  - 실시간 분석 엔드포인트
```

### 3. 중기 개선 (1-2개월)
```
🎯 고급 기능 추가
  - 연구 트렌드 예측 (TS-Unity)
  - 자동 이상 탐지 알림 (RAPID)
  - 논문 신뢰도 스코링 (Fake-News-Detection)

🎯 대시보드 개발
  - 수집 통계
  - 품질 지표
  - 성능 모니터링
```

---

## 📊 기술 비교표

| 프로젝트 | 주요 기술 | 입력 데이터 | 출력 | 복잡도 |
|---------|---------|----------|-----|--------|
| **TS-Unity** | Transformer, Autoformer | 시계열 데이터 | 예측/분류 | ⭐⭐⭐ |
| **RAPID** | BERT, 멀티프로세싱 | 텍스트 로그 | 이상 스코어 | ⭐⭐ |
| **Fake-News** | KoBERT, CNN | 한국어 텍스트 | 신뢰도 레이블 | ⭐⭐⭐ |
| **Text-Analytics** | 학습 자료 | - | 지식/가이드 | - |

---

## 🚀 다음 단계

### 1단계: 환경 설정
```bash
# 각 프로젝트의 의존성 분석
pip install -r requirements_ml.txt
```

### 2단계: 통합 모듈 개발
- 공통 유틸리티 추출
- 래퍼 함수 구현
- 설정 통합

### 3단계: 테스트 & 검증
- 단위 테스트 작성
- 통합 테스트
- 성능 벤치마크

### 4단계: 프로덕션 배포
- Docker 이미지 생성
- CI/CD 파이프라인 구성
- 모니터링 시스템 설치

---

## 📚 참고 자료

- **TS-Unity**: https://github.com/dsba-lab/TS-Unity
- **RAPID**: https://github.com/dsba-lab/RAPID
- **Fake-News-Detection**: https://github.com/dsba-lab/Fake-News-Detection-Dataset
- **Text-Analytics**: https://github.com/dsba-lab/Text-Analytics

---

## 🔧 기술 스택 정리

### 머신러닝 프레임워크
- PyTorch
- Transformers (HuggingFace)
- scikit-learn

### 한국어 처리
- KoBERT
- Konlpy
- Mecab

### 모니터링 & 실험
- Weights & Biases
- TensorBoard

### 개발 도구
- Docker
- Git

---

**작성일**: 2026-05-29
**분석 범위**: DSBA Lab 4개 주요 머신러닝 프로젝트
**적용 대상**: DBpia Crawler 프로젝트
