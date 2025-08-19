# Dobby: Zillow Time-on-Market Analyzer

이 저장소는 Zillow 부동산 데이터를 이용하여 매물의 판매 속도(Time-On-Market, TOM)를 예측하는 실험 코드입니다. 매물 설명과 속성 정보, 차별적 단어(discriminative words), 그리고 그 의미를 활용하여 LLM과 전통적 ML 모델을 비교합니다.

## 주요 기능
- **텍스트 전처리**: `text_preprocess.py`에서 Realtor가 작성한 설명을 정제합니다.
- **차별적 단어 추출**: `extract_words.py`에서 빠르게 혹은 느리게 팔리는 집에 자주 등장하는 단어를 수집합니다.
- **LLM 기반 분류**: `chain.py`에서 OpenAI 모델을 사용하여 'fast', 'moderate', 'slow' 세 단계로 TOM을 예측합니다.
- **전통적 ML 분류기**: `classifier.py`에 Logistic Regression, Random Forest, XGBoost 등을 구현해 성능을 비교합니다.
- **전체 파이프라인**: `main.py`에서 단어 추출, LLM 예측, 모델 평가, ML 학습을 순차적으로 수행합니다.

## 요구 사항
- Python 3.13+
- [Poetry](https://python-poetry.org/)

## 설치
```bash
poetry install
```
LLM 기능을 사용하려면 `.env` 파일에 `OPENAI_API_KEY`를 설정해야 합니다.

## 실행
기본 파이프라인 실행:
```bash
poetry run python main.py
```
`result/` 디렉터리에 각 임계값(threshold)에 대한 LLM 결과와 평가 리포트가 생성됩니다.

차별적 단어만 추출하려면:
```bash
poetry run python extract_words.py
```

## 데이터
- `dataset/2. zillow_cleaned.csv`: Zillow 매물 정보 및 판매 기간. (별도 제공)
- `dataset/word_counts/`: 각 도시/주택 유형별 단어 빈도 및 차별적 단어 목록.

## 라이선스
이 저장소의 라이선스는 명시되어 있지 않습니다. 사용 시 주의하세요.

## 연락처
- Maintainer: Hyeongchan Bae ([@Rlearnchan](https://github.com/Rlearnchan))
