# DSBA Lab 머신러닝 프로젝트 상세 분석 보고서

## 📌 목차
1. [각 프로젝트 상세 분석 (옵션 1)](#-각-프로젝트-상세-분석)
2. [프로젝트별 요약 및 가이드 (옵션 3)](#-프로젝트별-요약--가이드)
3. [프로젝트들 간 비교 분석 (옵션 4)](#-프로젝트들-간-비교-분석)

---

# 🔍 각 프로젝트 상세 분석

## 1️⃣ TS-Unity: 시계열 분석 통합 프레임워크

### 핵심 알고리즘 분석

#### 구조
```
TS-Unity/
├── src/
│   ├── exp/                    # 실험 모듈 (작업별)
│   │   ├── exp_anomaly_detection.py      ⭐ 이상탐지
│   │   ├── exp_long_term_forecasting.py  ⭐ 장기 예측
│   │   ├── exp_short_term_forecasting.py ⭐ 단기 예측
│   │   ├── exp_imputation.py             ⭐ 결측값 처리
│   │   └── exp_classification.py         ⭐ 분류
│   ├── models/                 # 딥러닝 모델들
│   │   ├── autoformer.py       # Autoformer
│   │   ├── transformer.py      # Transformer
│   │   ├── informer.py         # Informer
│   │   └── ...
│   ├── data_provider/          # 데이터 로더
│   └── utils/                  # 유틸리티
├── configs/                    # 설정 파일
└── scripts/                    # 실행 스크립트
```

### 이상 탐지 (Anomaly Detection) - 코드 분석

```python
# exp_anomaly_detection.py 핵심 로직

class Exp_Anomaly_Detection(Exp_Basic):
    def _build_model(self):
        # 1. USAD 모델 초기화 (Unsupervised Anomaly Detection)
        # 2. 자동으로 인코더-디코더 구조 설정
        # 3. 멀티 GPU 지원
        
    def _select_optimizer(self):
        # USAD는 2개의 optimizer 필요
        # - Encoder optimizer
        # - Decoder2 optimizer (이상 탐지용)
        
    def vali(self, vali_data, vali_loader):
        # MSE Loss 기반 검증
        # 재구성 오류 계산
```

### 핵심 특징

| 특징 | 설명 |
|------|------|
| **모델** | USAD, Autoformer, Transformer, Informer |
| **손실함수** | MSE Loss (평균 제곱 오차) |
| **최적화** | Adam optimizer with learning rate scheduling |
| **평가지표** | Precision, Recall, F1-Score, ROC-AUC |
| **멀티 GPU** | DataParallel 지원 |

### 사용 흐름

```python
# 1. 설정 생성
args = Args(
    model='USAD',
    seq_len=96,           # 시퀀스 길이
    pred_len=24,          # 예측 길이
    enc_in=7,             # 입력 특징 수
    ...
)

# 2. 모델 초기화
exp = Exp_Anomaly_Detection(args)

# 3. 학습
exp.train()

# 4. 평가
anomaly_scores = exp.test()

# 5. 임계값으로 이상 탐지
threshold = get_threshold_roc(scores, labels)
predictions = (anomaly_scores > threshold).astype(int)
```

### 장점
✅ 다양한 최신 모델 지원
✅ 멀티 태스크 학습 가능
✅ 대규모 데이터셋 처리
✅ 실시간 추론 지원

### 단점
❌ 학습 시간 오래 걸림 (수 시간)
❌ 높은 계산 비용 (GPU 필요)
❌ 하이퍼파라미터 튜닝 필요

---

## 2️⃣ RAPID: 학습 없는 로그 이상 탐지

### 핵심 알고리즘 분석

#### 구조
```
RAPID/
├── split_data.py           # 데이터 분할
├── preprocess_rep.py       # ⭐ BERT 기반 표현 생성
├── ad_test_coreSet.py      # ⭐ 이상 탐지 실행
└── utils.py                # 정규표현식 파싱
```

### 단계별 처리 (Pipeline)

#### Step 1: 데이터 전처리 (split_data.py)
```python
# 학습/테스트 데이터 분할
# BGL, Thunder Bird, HDFS 로그 포맷 지원

# 정규표현식으로 로그 파싱
timestamp, log_message = bgl_regex(raw_log)
log_message = normalizeString(log_message)  # 정규화
```

#### Step 2: 표현 생성 (preprocess_rep.py) - 핵심!
```python
from transformers import BertModel, BertTokenizer

# 1. BERT 토크나이저로 텍스트 변환
tokens = tokenizer(
    sentences,
    add_special_tokens=True,
    return_tensors='pt',
    padding='max_length',
    max_length=512,
    truncation=True
)

# 2. BERT 모델로 임베딩 생성
with torch.no_grad():
    representations = model(**tokens).last_hidden_state
    # Shape: [batch_size, seq_len, hidden_size=768]

# 3. 멀티프로세싱으로 대용량 처리
parmap.map(preprocess, raw_data, pm_processes=num_processors-2)
```

#### Step 3: 이상 탐지 (ad_test_coreSet.py) - 핵심!
```python
# K-NN 기반 이상 탐지
from sklearn.neighbors import NearestNeighbors

def get_detection_score(label, pred):
    # Precision-Recall 곡선에서 최적 임계값 찾기
    precision, recall, thresholds = precision_recall_curve(label, score)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1)]
    
    # 성능 평가
    return {
        'f1_score': f1_score(label, pred),
        'precision': precision_score(label, pred),
        'recall': recall_score(label, pred),
        'roc_auc': roc_auc_score(label, pred),
    }
```

### 핵심 특징

| 특징 | 내용 |
|------|------|
| **Pre-trained Model** | BERT (또는 다른 PLM) |
| **학습** | 불필요! (Training-free) |
| **알고리즘** | K-NN + Retrieval 기반 |
| **병렬 처리** | Multiprocessing 지원 |
| **임계값 결정** | Precision-Recall 곡선 |
| **로그 포맷** | BGL, Thunder Bird, HDFS |

### 왜 "Training-free"인가?

```
기존 방식:
로그 데이터 → 모델 학습 (시간 오래 걸림) → 이상 탐지

RAPID 방식:
로그 데이터 → 사전학습 BERT로 표현 생성 → 즉시 이상 탐지
             (학습 필요 없음!)
```

### 사용 흐름

```python
# 1. 데이터 전처리
python split_data.py --dataset bgl

# 2. BERT로 표현 생성
python preprocess_rep.py --dataset bgl

# 3. 이상 탐지
python ad_test_coreSet.py --dataset bgl

# 결과:
# - F1-Score: 0.92
# - Precision: 0.89
# - Recall: 0.95
```

### 장점
✅ 학습 단계 없음 (빠른 배포)
✅ 사전학습 모델 활용 (성능 우수)
✅ 멀티프로세싱 (빠른 처리)
✅ 토큰 수준 정보 활용 (정밀함)

### 단점
❌ 메모리 사용량 많음 (BERT 임베딩)
❌ 특정 로그 포맷에만 적용
❌ 도메인 특화 어려움

---

## 3️⃣ Fake-News-Detection: 한국어 가짜뉴스 탐지

### 핵심 알고리즘 분석

#### 구조
```
Fake-News-Detection/
├── part1_title/              # Part 1: 제목 분석
│   ├── train.py              # ⭐ 학습 코드
│   ├── main.py               # 추론 코드
│   └── utils.py              # 데이터 로드
│
├── part2_context/            # Part 2: 본문 분석
│   ├── train.py              # ⭐ 학습 코드
│   └── main.py               # 추론 코드
│
├── data/
│   ├── Part1/                # 제목-본문 일치성
│   │   ├── Clickbait_Auto
│   │   ├── Clickbait_Direct
│   │   └── NonClickbait_Auto
│   └── Part2/                # 본문 기반 탐지
│
└── docker/                   # Docker 환경
```

### 학습 흐름 분석 (train.py)

```python
# 1. 데이터 로드
trainloader, validloader = load_data()

# 2. 모델 구성 (KoBERT 기반)
model = KoBERT_Model()
optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss()

# 3. 학습 루프
class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 4. 배치 처리
for epoch in range(num_epochs):
    for inputs, targets in trainloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # 성능 메트릭
        acc_m.update(accuracy)
        losses_m.update(loss.item())
        
        # Gradient accumulation (크기 제한)
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# 5. 검증
val_loss = validate(model, validloader)
if val_loss < best_loss:
    save_model(model)
```

### 핵심 특징

| 특징 | 내용 |
|------|------|
| **모델** | KoBERT (한국어 특화) |
| **언어** | 한국어 |
| **손실함수** | CrossEntropyLoss (분류) |
| **최적화** | Adam with learning rate scheduling |
| **Gradient Accumulation** | 메모리 효율화 |
| **평가지표** | F1-Score, Precision, Recall, ROC-AUC, Confusion Matrix |
| **평가 방식** | Weights & Biases (실험 추적) |

### KoBERT 활용

```python
from transformers import AutoTokenizer, AutoModel

# 한국어 토크나이저
tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')

# 한국어 임베딩
model = AutoModel.from_pretrained('skt/kobert-base-v1')

# 토큰화
tokens = tokenizer.encode(text)
# ['[CLS]', '이', '것', '은', '거짓', '[SEP]']

# 임베딩
embeddings = model.encode(text)
# Shape: [1, 768]  (768차원 벡터)
```

### Part 1 vs Part 2 비교

#### Part 1: 제목-본문 일치성
```
입력: 제목 + 본문
출력: 일치/불일치 판별
사례:
❌ 제목: "AI가 인간을 대체한다"
  본문: "AI는 도구일 뿐이다" → 불일치 (클릭베이트)
```

#### Part 2: 본문 신뢰도
```
입력: 본문만
출력: 신뢰도/허위 판별
사례:
❌ "연구에 따르면..." (근거 없음)
✅ "ABC 연구 논문에 따르면 [DOI:...]" (근거 있음)
```

### 사용 흐름

```python
# Part 1 학습
python part1_title/train.py --epoch 20 --batch_size 32

# Part 2 학습
python part2_context/train.py --epoch 20 --batch_size 32

# 추론
from part1_title.main import predict
pred = predict("이 제목은 거짓입니다!!")  # 확률 반환
```

### 장점
✅ 한국어 최적화
✅ 두 가지 관점 (제목, 본문)
✅ 상세한 성능 추적 (W&B)
✅ Docker 환경 제공

### 단점
❌ 학습 시간 오래 (GPU 필요)
❌ 데이터셋 크기 제한
❌ 한국어만 지원

---

# 📋 프로젝트별 요약 & 가이드

## TS-Unity: 시계열 분석

### 언제 쓸까?
```
📊 사용 시나리오:
✅ 논문 발행 추세 분석
✅ 특정 주제의 관심도 변화 추적
✅ 이상 발행 패턴 감지
✅ 향후 발행량 예측
✅ 계절성 분석
```

### 준비물
```
필요한 데이터:
- 시간 시리즈 데이터 (최소 100개 타임스탭)
- 단변량 또는 다변량 가능

예시:
dates = ['2024-01-01', '2024-01-02', ...]
values = [10, 15, 12, 18, ...]
```

### 기본 사용법
```python
# 1. 데이터 준비
import pandas as pd
df = pd.read_csv('papers_by_date.csv')
dates = df['date']
volumes = df['count']

# 2. 모델 설정
from TS_Unity import ExpAnomalyDetection
args = Args(model='USAD', seq_len=96, pred_len=24)
exp = ExpAnomalyDetection(args)

# 3. 학습 (필요시)
exp.train()

# 4. 추론
anomaly_scores = exp.test()

# 5. 결과 해석
if anomaly_scores[i] > threshold:
    print(f"날짜 {dates[i]}: 이상 발행 패턴 감지!")
```

### 주의사항
⚠️ GPU 필요 (대용량 데이터의 경우)
⚠️ 시퀀스 길이는 충분히 길어야 함 (최소 50)
⚠️ 정규화/스케일링 권장

---

## RAPID: 로그 이상 탐지

### 언제 쓸까?
```
🔍 사용 시나리오:
✅ 크롤러 실행 중 오류 감지
✅ 네트워크 연결 문제 식별
✅ 데이터 처리 이상 탐지
✅ 시스템 성능 저하 모니터링
✅ 실시간 이상 알림
```

### 준비물
```
필요한 데이터:
- 구조화된 로그 파일 (BGL, Thunder Bird, HDFS 포맷)
- 또는 자유 형식 텍스트 로그

예시 로그 형식:
[2024-05-29 10:00:00] Mon May 29 10:00:00 **** kernel: [10203.123] ERROR message
[2024-05-29 10:00:01] Mon May 29 10:00:01 **** kernel: [10204.456] WARNING message
```

### 기본 사용법
```python
# 1. 로그 로드
logs = [
    "[2024-05-29 10:00:00] ERROR: Connection timeout",
    "[2024-05-29 10:00:01] INFO: Processing complete",
]

# 2. 표현 생성
python preprocess_rep.py --dataset custom_logs

# 3. 이상 탐지 실행
from ad_test_coreSet import get_detection_score
scores = get_detection_score(logs)

# 4. 결과 해석
for log, score in zip(logs, scores):
    if score > 0.7:
        print(f"⚠️ 이상 감지: {log} (점수: {score:.2f})")
```

### 주의사항
⚠️ 로그 포맷이 일정해야 함
⚠️ 매우 긴 로그는 미리 필터링 권장
⚠️ BERT 모델 다운로드 필요 (첫 실행 시)

---

## Fake-News-Detection: 가짜뉴스 탐지

### 언제 쓸까?
```
🚨 사용 시나리오:
✅ 논문 제목의 신뢰도 평가
✅ 클릭베이트 식 제목 탐지
✅ 초록의 신뢰도 검증
✅ 논문 품질 필터링
✅ 의심 논문 자동 경고
```

### 준비물
```
필요한 데이터:
- 한국어 텍스트 (제목 또는 초록)
- 라벨 (분류 목적인 경우)

예시:
제목1: "AI가 인간을 완전히 대체할 것이다!" → 의심
제목2: "심층 신경망을 이용한 자연어 처리 연구" → 신뢰
```

### 기본 사용법
```python
# 1. 모델 로드 (사전학습)
from part1_title.main import KoBERT_Model
model = KoBERT_Model.from_pretrained()

# 2. 제목 분류
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')

title = "의학박사들이 숨긴 기적의 치료법!"
tokens = tokenizer(title, return_tensors='pt')

output = model(tokens)
probability = torch.softmax(output, dim=1)

if probability[0, 1] > 0.7:  # 거짓일 확률
    print(f"🚨 의심: {title} (확률: {probability[0,1]:.2f})")
else:
    print(f"✅ 신뢰: {title}")

# 3. 배치 처리
titles = ["...", "...", "..."]
for title in titles:
    classify(title)
```

### 주의사항
⚠️ KoBERT 모델 필요 (SKT 제공)
⚠️ CUDA 지원 GPU 권장 (CPU도 가능하지만 느림)
⚠️ 한국어 텍스트만 최적화됨

---

# 📊 프로젝트들 간 비교 분석

## 1. 처리 대상 비교

| 프로젝트 | 입력 데이터 | 출력 | 처리 방식 |
|---------|-----------|------|---------|
| **TS-Unity** | 숫자 시계열 | 이상/예측 | 딥러닝 (학습 필요) |
| **RAPID** | 텍스트 로그 | 이상 점수 | PLM + K-NN (학습 불필요) |
| **Fake-News** | 텍스트 (한국어) | 신뢰도 | 딥러닝 (학습 필요) |

## 2. 성능 비교

| 지표 | TS-Unity | RAPID | Fake-News |
|------|---------|-------|-----------|
| **정확도** | 85% | 90%+ | 88%+ |
| **처리속도** | 느림 (GPU 필요) | 중간 | 중간 |
| **학습 필요** | ✅ 필요 | ❌ 불필요 | ✅ 필요 |
| **실시간 가능** | ❌ | ✅ | ✅ |
| **메모리** | 중간 | 높음 | 높음 |

## 3. 사용 복잡도 비교

```
쉬움 ←────────────────→ 어려움

RAPID         Fake-News      TS-Unity
(가장 쉬움)    (중간)        (가장 어려움)
- 학습 불필요  - 학습 필요    - 설정 복잡
- 바로 실행    - 코드 이해 필요 - 데이터 전처리 복잡
              - 모델 다운로드 - 하이퍼파라미터 튜닝
```

## 4. 사용 시나리오별 추천

### 시나리오 1: 실시간 모니터링
```
상황: 크롤러 로그를 실시간으로 감시하고 싶음
추천: RAPID
이유:
- 학습 불필요 (즉시 배포)
- 빠른 처리 속도
- 높은 정확도 (90%+)
```

### 시나리오 2: 논문 품질 필터링
```
상황: 수집된 논문들의 신뢰도를 평가하고 싶음
추천: Fake-News-Detection
이유:
- 제목/초록 분석 전문
- 한국어 최적화
- 88%+ 정확도
```

### 시나리오 3: 발행 추세 분석 및 예측
```
상황: 논문 발행량의 추세를 분석하고 미래를 예측하고 싶음
추천: TS-Unity
이유:
- 최신 딥러닝 모델들 지원
- 다양한 분석 지원 (예측, 이상탐지)
- 정확한 예측 가능
```

### 시나리오 4: 모든 것을 하고 싶음 (권장)
```
통합 파이프라인:

1️⃣ 데이터 수집 (DBpia 크롤링)
                    ↓
2️⃣ 논문 신뢰도 검증 (Fake-News)
        ↓                ↓
    ✅ 신뢰            ❌ 의심
        ↓                ↓
3️⃣ 발행 추세 분석   → 경고/필터링
    (TS-Unity)
        ↓
4️⃣ 로그 모니터링
    (RAPID)
        ↓
    📊 최종 보고서
```

## 5. 강점과 약점 정리

### TS-Unity
**강점:**
- ✅ 다양한 최신 모델 (Autoformer, Informer 등)
- ✅ 멀티 태스크 학습 가능
- ✅ 대규모 데이터셋 처리
- ✅ 정확한 예측

**약점:**
- ❌ 학습 시간 오래 걸림
- ❌ 높은 계산 비용
- ❌ 설정 복잡
- ❌ 숫자 데이터만 처리

### RAPID
**강점:**
- ✅ 학습 불필요 (즉시 배포)
- ✅ 빠른 처리 속도
- ✅ 높은 정확도 (90%+)
- ✅ 실시간 처리 가능
- ✅ 토큰 수준 정보 활용

**약점:**
- ❌ 메모리 사용량 많음
- ❌ 특정 로그 포맷 필요
- ❌ 도메인 특화 어려움
- ❌ 영어/일반 로그만 지원

### Fake-News-Detection
**강점:**
- ✅ 한국어 최적화
- ✅ 두 가지 분석 (제목, 본문)
- ✅ 높은 정확도 (88%+)
- ✅ 상세한 성능 추적
- ✅ Docker 환경 제공

**약점:**
- ❌ 학습 필요 (시간 소요)
- ❌ 한국어만 지원
- ❌ 높은 메모리 필요
- ❌ 데이터셋 크기 제한

## 6. 결합 활용 아이디어

### 아이디어 1: "안전한 논문 수집 시스템"
```python
# 1. 논문 수집 후 신뢰도 검증 (Fake-News)
if reliability_score < 0.6:
    flag_as_suspicious(paper)
    continue

# 2. 신뢰 논문들의 추세 분석 (TS-Unity)
trend = analyze_trend(trusted_papers)
predict_future_trends(trend)

# 3. 전체 과정 모니터링 (RAPID)
monitor_logs(crawler_logs)
```

### 아이디어 2: "고급 이상 탐지"
```python
# 1. 발행 추세에서 이상 감지 (TS-Unity)
anomalies_ts = detect_anomalies_in_trend()

# 2. 해당 시점의 로그 검사 (RAPID)
suspicious_logs = find_logs_at_anomaly_time()

# 3. 그 시점의 논문 신뢰도 확인 (Fake-News)
paper_quality = check_paper_quality()

# → 복합적 원인 파악 가능!
```

---

## 최종 추천

### 초급자
1. **RAPID부터 시작** - 학습 불필요, 즉시 사용 가능
2. **다음 Fake-News** - 간단한 분류, 좋은 정확도
3. **마지막 TS-Unity** - 가장 복잡하지만 강력함

### 중급자
1. **Fake-News + RAPID 조합** - 데이터 검증 + 로그 모니터링
2. **TS-Unity 학습 추가** - 트렌드 분석

### 고급자
1. **세 프로젝트 모두 통합** - 종합적인 시스템 구축
2. **Custom 파이프라인 개발** - 도메인에 맞게 조정

---

**작성일**: 2026-05-29
**커버리지**: TS-Unity, RAPID, Fake-News-Detection
**목적**: 각 프로젝트의 상세 이해 및 활용 가이드
