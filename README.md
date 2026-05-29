## DBpia 디비피아 크롤러
국내 논문 서지정보 사이트 DBpia내의 논문제목, 저자, 퍼블리셔, 저널명, 볼륨, 날짜, 초록의 내용을 가져올수 있는 프로그램
### 설치
DBpia 디비피아 크롤러는 파이썬3를 통해 작성되었으며 BeautifulSoup과 셀레늄 패키지를 활용하였습니다.<br><br>
먼저 해당 프로그램을 git clone을 통해 다운로드 해줍니다.
```
$git clone https://github.com/chanhee-kang/DBpia_crawler.git
```
그후, 해당 프로그램을 사용하기 위해서는 자신의 크롬의 버전과 일치하는 크롬드라이버가 필요합니다.<br><br>
크롬드라이버 다운로드는 다음 링크를 통해 설치가 가능합니다. [https://chromedriver.chromium.org/downloads] <br><br>
또한, 아나콘다(Anaconda)환경에서 제작되어서 아나콘다내 파이썬 내장 패키지외 추가 패키지를 터미널에서 pip 명령어를통해 설치 해주어야 합니다
```
$pip install selenium
$pip install beautifulsoup
```

### 실행
아래 코드에 DBpia 내에서 크롤링할 검색어 설정, 날짜, 다운로드 받으신 크롬드라이버의 경로 설정을 해줍니다.
```
searchQ = "검색어 삽입"
startYear = 크롤링 시작 벙위 설정
endYear = 크롤링 마침 벙위 설정

print("start crawling..")

path = 'chromedriver 경로 지정'
```
코드 밑단 부분에 크롤링 결과가 저장될 경로를 삽입시켜줍니다.
```
fName = "{}_{}_{}.csv".format(searchQ, startYear, endYear)
```
코드를 실행시켜 주시면 Selenium을 통한 크롤링이 진행됩니다.

### 보안점
멀티 쓰레딩 추가

### 머신러닝 모듈 (DSBA Lab Integration)

DBpia_crawler는 이제 DSBA Lab의 주요 연구 성과를 통합한 머신러닝 모듈을 제공합니다.

#### 주요 기능

**1. 시계열 분석 (TS-Unity Framework)**
```python
from ml_modules import TimeSeriesAnalyzer

analyzer = TimeSeriesAnalyzer()
# 논문 발행 추세 분석, 이상 탐지, 예측
```

**2. 로그 이상 탐지 (RAPID Framework)**
```python
from ml_modules import LogAnomalyDetector

detector = LogAnomalyDetector()
# 크롤러 로그 모니터링, 오류 감지
```

**3. 텍스트 신뢰도 분석 (Fake-News-Detection)**
```python
from ml_modules import FakeNewsDetector

classifier = FakeNewsDetector()
# 논문 제목/초록 신뢰도 평가
```

#### 설치

```bash
# 기본 설치
pip install -r requirements_ml.txt

# 고급 기능 (선택사항)
pip install transformers torch
```

#### 사용 예제

```python
from ml_modules import TimeSeriesAnalyzer, LogAnomalyDetector, FakeNewsDetector

# 수집된 논문 분석
papers = load_papers()  # 크롤러로부터 수집

# 논문 신뢰도 검증
classifier = FakeNewsDetector()
for paper in papers:
    result = classifier.classify_title(paper['title'])
    
# 수집 패턴 분석
analyzer = TimeSeriesAnalyzer()
trend_analysis = analyzer.analyze_publication_trends(dates, volumes)

# 로그 모니터링
log_detector = LogAnomalyDetector()
log_results = log_detector.analyze_logs(crawler_logs)
```

더 자세한 정보는 [ml_modules/README.md](ml_modules/README.md)를 참고하세요.

### Contact
If you have any requests, please contact: [https://ck992.github.io/](https://ck992.github.io/).

