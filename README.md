# universe
매칭 시스템입니다. 매칭의 전과정을 포함하였습니다. (전처리, 파이프 라인, 모델 평가) 

# Matching Universe

AI 기반 채용 후보자 매칭 파이프라인입니다. JD(채용공고)와 CV(후보자 이력서)를 입력받아, BM25/TF-IDF 키워드 검색 → BGE-M3 임베딩 검색 → BGE-Reranker-v2-M3 정밀 재채점 → Gemini 기반 추천 해설까지 4단계 매칭 구조로 후보자 순위를 산출합니다.

```
#ai-recruitment  #candidate-matching  #bge-m3  #cross-encoder  #gemini  #nlp  #korean
```

---

## Table of contents

- [Matching Universe](#matching-universe)
- [Guide](#guide)
  - [Usage guide](#usage-guide)
  - [Parameter naming](#parameter-naming)
- [Setup](#setup)
  - [Python version](#python-version)
  - [Requires](#requires)
- [Pipeline](#pipeline)
  - [Taxonomy 표준화](#taxonomy-표준화)
  - [매칭용 텍스트 생성](#매칭용-텍스트-생성)
  - [1차 Retrieval — BM25 + TF-IDF](#1차-retrieval--bm25--tf-idf)
  - [1차 Retrieval — BGE-M3 Embedding](#1차-retrieval--bge-m3-embedding)
  - [Cross-Encoder 정밀 재채점](#cross-encoder-정밀-재채점)
  - [AI 태그 Overlap + 역할 패널티](#ai-태그-overlap--역할-패널티)
  - [점수 정규화 및 final\_score 계산](#점수-정규화-및-final_score-계산)
  - [Ground Truth 평가](#ground-truth-평가)
  - [LLM 해설 (Optional)](#llm-해설-optional)
  - [결과 저장](#결과-저장)
- [Output files](#output-files)
- [Notes](#notes)
- [Roadmap](#roadmap)

---

## Guide

Matching Universe는 한국어 AI 직군 채용(Analyst · Engineer · Researcher · Scientist)을 위해 설계된 후보자 매칭 파이프라인입니다.
비지도 기반의 통계 키워드 매칭과 사전학습 언어모델 기반의 의미 검색을 함께 사용하기 때문에, 새로운 기술 용어나 프로젝트명처럼 Taxonomy에 등록되지 않은 단어도 임베딩을 통해 유사 후보를 찾아낼 수 있습니다.

핵심 설계 원칙은 두 가지입니다.

첫째, **전체 후보를 LLM에 넘기지 않습니다.** BM25/TF-IDF → 임베딩 검색 → Cross-Encoder의 3단계 깔때기로 점점 좁혀가며, 가장 계산 비용이 높은 LLM 해설은 최종 Top-N에게만 적용합니다.

둘째, **모든 점수 변경의 효과를 Ground Truth로 측정합니다.** 가중치를 감으로 바꾸는 대신, 실제 정답 라벨(`relevance_grade`)을 기준으로 Recall@K · Precision@K · MRR · NDCG@K를 계산해 비교합니다.

### Usage guide

노트북의 실행 흐름은 셀 번호 순서를 따릅니다. 처음 실행해 보는 경우, `Cell 01`에서 아래 두 값만 조정하고 **Cell → Run All** 하면 됩니다.

```python
# Cell 01 | 경로 및 실행 설정

# 매칭을 돌릴 JD 목록 — 처음에는 1~2개만 지정해 빠르게 확인하세요.
MANUAL_TARGET_JD_IDS = [191]

# Cross-Encoder 정밀 재채점을 적용할 상위 후보 수
# CPU 환경에서는 줄일수록 속도가 빨라집니다.
TOP_N_CROSS_ENCODER = 100
```

Ground Truth 평가셋(`data/10_evaluation/ground_truth_matches.xlsx`)이 준비된 경우, `TARGET_JD_IDS`는 정답 파일에 등록된 JD 목록으로 자동 결정됩니다. 파일이 없으면 빈 템플릿만 자동 생성하고 중단되므로, 정답 라벨을 채운 뒤 다시 실행하세요.

### Parameter naming

Cell 01에 모든 주요 설정이 모여 있습니다. 설정 변수는 아래 규칙으로 명명합니다.

- `USE_*` : 기능 토글 (`True` / `False`)
- `TOP_N_*` : 단계별 상위 후보 수 제한
- `*_WEIGHT` : 최종 `final_score` 계산 시 각 점수의 비중
- `*_PATH` : 입력/출력 파일 절대 경로
- `WEIGHTS` : `column_score` 내부의 컬럼별 가중치 dict
- `RETRIEVAL_COLUMN_WEIGHTS` : 1차 Retrieval 깔때기 단계의 컬럼별 가중치 dict

가중치는 두 레이어로 분리되어 있습니다.

```python
# 1차 Retrieval 깔때기용 (전체 865명 → Top-N 좁히기)
RETRIEVAL_COLUMN_WEIGHTS = {
    "skill": 0.60,
    "task": 0.25,
    "role": 0.10,
    "domain": 0.05,
}

# 최종 column_score 내부 정밀 가중치 (Top-N 후보 정밀 재채점)
WEIGHTS = {
    "skill_score": 0.35,
    "task_score": 0.25,
    "domain_score": 0.10,
    "role_score": 0.15,
    "ai_skill_overlap_score": 0.15,
}

# final_score = retrieval_score_norm × RETRIEVAL_SCORE_WEIGHT
#             + column_score         × COLUMN_SCORE_WEIGHT
RETRIEVAL_SCORE_WEIGHT = 0.25
COLUMN_SCORE_WEIGHT    = 0.75
```

---

## Setup

### Python version

Python 3.10 이상을 지원합니다. 개발 및 테스트는 3.10 / 3.11 환경에서 진행합니다.
Python 3.9 이하에서는 동작을 보장하지 않습니다.

### Requires

```text
numpy
pandas
scikit-learn
openpyxl
sentence-transformers   # BGE-M3, BGE-Reranker-v2-M3 로드에 필요
google-generativeai     # Gemini API (LLM 해설 기능을 쓸 때만 필요)
jupyter
ipykernel
```

아래 순서로 설치합니다.

```bash
# 저장소 클론
git clone https://github.com/doctorSong1/universe.git
cd universe

# 가상환경 생성 및 활성화
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# Jupyter 커널 등록 (노트북에서 가상환경을 선택할 수 있도록)
python -m ipykernel install --user --name matching-universe --display-name "Python (matching-universe)"
```

GPU(CUDA)는 필수가 아닙니다. CPU만으로도 동작하지만, Cross-Encoder 재채점 단계는 GPU 사용 시 현저히 빨라집니다. GPU 환경이라면 `sentence-transformers` 설치 전에 [PyTorch 공식 가이드](https://pytorch.org/get-started/locally/)에서 CUDA 버전에 맞는 `torch`를 먼저 설치하는 것을 권장합니다.

#### Gemini API Key 설정 (LLM 해설 기능 사용 시)

기본값은 `USE_LLM_EVALUATION = False`로 비활성화 상태입니다. LLM 해설 기능을 켜려면 아래와 같이 API Key를 설정합니다.

```bash
# macOS / Linux
export GEMINI_API_KEY="발급받은_API_KEY"

# Windows (PowerShell)
$env:GEMINI_API_KEY="발급받은_API_KEY"
```

또는 `Cell 02`에서 직접 입력할 수도 있습니다. 단, 이 경우 커밋 전에 반드시 값을 삭제해야 합니다.

```python
# Cell 02 | API Key 입력부 (커밋 전 삭제 필수)
os.environ["GEMINI_API_KEY"] = "여기에_API_KEY_입력"
```

#### 데이터 파일 준비

CV 데이터는 개인정보 보호를 위해 이 저장소에 포함되지 않습니다. JD 데이터와 임베딩 파일은 별도 공유 예정입니다. 아래 경로 구조에 맞게 파일을 배치하면 노트북이 경로를 자동으로 인식합니다.

```text
universe/
├── configs/
│   └── taxonomy_stopwords.yaml          # 불용어 사전 (필수)
├── data/
│   ├── 05_profiles/
│   │   └── candidate_profile_jo_all_FINAL5.xlsx   ← CV Master (비공개, 별도 준비)
│   ├── 07_embeddings/
│   │   ├── cv_embeddings_only_bge_m3_10.npz        ← CV 임베딩 벡터 (별도 준비)
│   │   ├── cv_embedding_meta_bge_m3_10.csv          ← CV 임베딩 메타
│   │   └── jd_embeddings_bge_m3.npy                ← JD 임베딩 벡터 (별도 준비)
│   ├── 10_jd/
│   │   └── jd_dataset_613_filled.csv               ← JD Master (필수)
│   ├── 10_evaluation/
│   │   └── ground_truth_matches.xlsx               ← 정답 평가셋 (평가 실행 시 필수)
│   └── 11_taxonomy/
│       └── *_taxonomy_jo.xlsx                      ← Taxonomy 사전 (선택)
└── src/
    └── normalization/
        └── taxonomy_text.py
```

프로젝트 루트 탐지는 `Cell 01`의 `find_project_root()` 함수가 자동으로 처리합니다. 저장소를 클론한 폴더 안에서 Jupyter를 실행하면 별도 설정 없이 경로를 인식합니다.

---

## Pipeline

### Taxonomy 표준화

`Cell 05 ~ 09`에서 진행합니다. JD와 CV의 원문 텍스트 컬럼(Skill, Career, Career_Description, Position 등)을 표준 어휘(`*_standard`)로 변환합니다.

Taxonomy 연결 우선순위는 다음과 같습니다.

```text
1순위: data/11_taxonomy/*_taxonomy_jo.xlsx (외부 Excel)
2순위: 노트북 내부 기본 Dictionary (fallback)
```

아래는 외부 Taxonomy 파일을 로드하는 예시입니다.

```python
from src.normalization.taxonomy_text import load_stopwords, is_stopword, normalize_token

TAXONOMY_STOPWORDS, TAXONOMY_STOPWORD_PATTERNS = load_stopwords(STOPWORDS_PATH)

# taxonomy_jo.xlsx 형식
# | term | standard | category |
# |------|----------|----------|
# | LLM  | LLM      | skill    |
# | RAG  | RAG      | skill    |
```

표준화 결과로 생성되는 컬럼:

```text
CV: cv_skill_standard, cv_task_standard, cv_domain_standard, cv_role_standard
JD: jd_skill_standard, jd_task_standard, jd_domain_standard, jd_role_standard
```

컬럼이 원본 파일에 없어도 노트북이 자동으로 빈 문자열 컬럼을 생성하므로 중단 없이 진행됩니다.

---

### 매칭용 텍스트 생성

`Cell 10`에서 진행합니다. 1차 검색용 `full_text`와 정밀 비교용 영역별 텍스트를 동시에 생성합니다.

```python
# CV 텍스트 생성 예시 (Cell 10)
def build_cv_texts(row):
    # full_text: 1차 Retrieval 전용 — 전체 맥락을 한 문자열에 담습니다
    cv_full_text = " ".join(filter(None, [
        row.get("Skill"),
        row.get("Career"),
        row.get("Career_Description"),
        row.get("cv_skill_standard"),
        row.get("cv_task_standard"),
        ...
    ]))

    # 영역별 텍스트: Cross-Encoder 정밀 비교 전용
    cv_skill_text  = " ".join(filter(None, [row.get("Skill"), row.get("cv_skill_standard")]))
    cv_task_text   = " ".join(filter(None, [row.get("Career_Description"), row.get("cv_task_standard")]))
    cv_domain_text = row.get("cv_domain_standard", "")
    cv_role_text   = " ".join(filter(None, [row.get("Position"), row.get("cv_role_standard")]))
    ...
```

---

### 1차 Retrieval — BM25 + TF-IDF

`Cell 11`에서 진행합니다. 외부 패키지 없이 BM25를 직접 구현합니다. BM25 점수와 TF-IDF cosine 점수를 각각 계산한 뒤 `RETRIEVAL_COLUMN_WEIGHTS`를 적용해 합산합니다.

```python
# BM25 파라미터 (Cell 11 기본값)
BM25_K1 = 1.5
BM25_B  = 0.75

# 1차 Retrieval 컬럼 가중치
RETRIEVAL_COLUMN_WEIGHTS = {
    "skill": 0.60,   # Skill 텍스트 중심
    "task": 0.25,    # Career_Description 보완
    "role": 0.10,
    "domain": 0.05,
}
```

결과로 `keyword_baseline_score`가 생성됩니다. 이 점수는 이후 `final_score`의 한 요소로 사용됩니다.

---

### 1차 Retrieval — BGE-M3 Embedding

`Cell 12`에서 진행합니다. `USE_EMBEDDING_RETRIEVAL = True`일 때 활성화됩니다.

CV 임베딩(`.npz`)과 JD 임베딩(`.npy`)을 로드한 뒤, 코사인 유사도로 JD별 상위 후보를 추립니다.

```python
USE_EMBEDDING_RETRIEVAL = True    # Cell 01에서 설정
TOP_N_RETRIEVAL = 865             # 1차 검색에서 JD별로 끌어올 최대 후보 수
                                  # 전체 후보(865명)로 설정하면 Retrieval 필터링을 사용하지 않는 것과 동일

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # 1024차원
```

임베딩 파일이 없거나 `USE_EMBEDDING_RETRIEVAL = False`이면, BM25/TF-IDF 키워드 점수만으로 1차 후보를 구성합니다.

`Cell 13`에서 키워드 점수와 임베딩 점수를 합쳐 `pair_df`(JD-CV 후보 pair 테이블)를 만듭니다. 이후 모든 정밀 재채점은 이 `pair_df`를 입력으로 사용합니다.

---

### Cross-Encoder 정밀 재채점

`Cell 16`에서 진행합니다. `pair_df`의 JD별 상위 `TOP_N_CROSS_ENCODER`명에게만 Cross-Encoder를 적용해 계산 비용을 제어합니다.

```python
USE_CROSS_ENCODER        = True   # Cell 01에서 설정
TOP_N_CROSS_ENCODER      = 100    # JD별 Cross-Encoder 적용 대상 상위 N명
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 4개 영역 각각 독립적으로 재채점
# (skill_text, task_text, domain_text, role_text)
# task/domain은 입력 길이를 512 토큰으로 제한합니다.
```

`USE_CROSS_ENCODER = False`이면 Cross-Encoder 대신 TF-IDF cosine similarity(`Cell 14`)로 대체합니다.

Cross-Encoder 결과로 생성되는 컬럼:

```text
skill_score, task_score, domain_score, role_score
```

---

### AI 태그 Overlap + 역할 패널티

`Cell 17 ~ 18`에서 진행합니다.

**AI 태그 Overlap** (`Cell 17`): `"LLM, RAG, Cloud"` 같은 태그 문자열끼리 겹치는 비율을 계산합니다.

```python
# JD 태그 중 CV 태그가 몇 개 포함되는지 비율 계산
# 예: JD = "LLM, RAG, Agent"  /  CV = "LLM, RAG"  →  overlap = 0.67
ai_skill_overlap_score = len(jd_tags & cv_tags) / len(jd_tags)
```

**역할 패널티** (`Cell 18`): Engineer JD에 Researcher/Scientist 성향의 후보가 과도하게 상위로 올라오는 현상을 보정합니다.

```python
def role_mismatch_penalty(row):
    # jd_role_text와 cv_role_text의 키워드를 비교해
    # 역할 불일치가 크면 최대 30%까지 점수를 감산합니다.
    ...
```

---

### 점수 정규화 및 final\_score 계산

`Cell 19`에서 진행합니다. 모델·방식마다 스케일이 다른 점수를 JD 단위로 min-max 정규화한 뒤 가중합합니다. 같은 JD 안에서 후보 간 상대 비교가 가능한 점수로 만드는 것이 목적입니다.

```python
# 1단계: JD별 min-max 정규화
# retrieval_score → retrieval_score_norm
# keyword_baseline_score → keyword_baseline_score_norm
# column_score 내부 컬럼 (skill_score 등) → 각각 정규화

# 2단계: column_score 합산
column_score = sum(score_col × weight for score_col, weight in WEIGHTS.items())

# 3단계: final_score
final_score = (retrieval_score_norm × RETRIEVAL_SCORE_WEIGHT
             + column_score         × COLUMN_SCORE_WEIGHT)

# 주요 0~1 점수는 100점 환산 컬럼(*_100)도 함께 생성합니다.
# final_score_100, skill_score_100, task_score_100, ...
```

대시보드 육각형 차트용 6축 지표는 `Cell 20`에서 생성합니다.

```text
Radar 6축:
  Skill       — 기술 스택 적합도 (skill_score 기반)
  Task        — 업무 경험 적합도 (task_score 기반)
  Domain      — 산업 도메인 적합도 (domain_score 기반)
  Role        — 직무 역할 적합도 (role_score 기반)
  Experience  — 경력 연수 기반 점수
  Soft Skill  — 협업·커뮤니케이션 등 소프트스킬 점수
```

---

### Ground Truth 평가

`Cell 21 ~ 25`에서 진행합니다.

`data/10_evaluation/ground_truth_matches.xlsx`가 유일한 정답 기준입니다. 파일이 없으면 빈 템플릿을 자동 생성하고 안내 메시지와 함께 중단됩니다.

**필수 컬럼**

```text
jd_id           — JD 식별자 (JD Master의 jd_id와 일치)
candidate_id    — 후보자 식별자 (CV Master의 candidate_id와 일치)
relevance_grade — 0: 비관련 / 1: 보류 / 2: 면접 추천 / 3: 최우선 후보
```

**계산 지표**

```text
Recall@K     : 정답 중 상위 K개 안에 포함된 비율
Precision@K  : 상위 K개 중 실제 정답(relevance_grade ≥ 2)인 비율
MRR          : 첫 번째 정답이 등장한 순위의 역수 평균
NDCG@K       : relevance_grade를 graded relevance로 사용한 순위 품질 지표
```

`Cell 22`의 **가중치 비교**와 `Cell 23`의 **Grid Search**는 이 지표를 기준으로 최적 가중치 조합을 자동 탐색합니다. 결과는 `data/10_evaluation/weight_config_validation_latest.xlsx`와 `grid_search_hybrid_v26.xlsx`에 저장됩니다.

---

### LLM 해설 (Optional)

`Cell 27 ~ 42`에서 진행합니다. 기본값은 모두 `False`로 비활성화 상태입니다.

```python
# Cell 01에서 아래 값을 True로 변경
USE_LLM_EVALUATION      = True  # Gemini API 전반 활성화
USE_GEMINI              = True
USE_LLM_EXPLANATION     = True  # 추천 요약 / 강점 / Gap / 면접 질문
USE_HARDSKILL_EVIDENCE_LLM = True   # 하드스킬 Evidence (V1/V2 판정)
USE_SOFTSKILL_LLM           = True  # 소프트스킬 S1~S5 (0/1 + 근거 인용)
USE_CAREER_PATH_LLM         = True  # 커리어 패스 타임라인 요약

GEMINI_MODEL = "gemini-2.5-flash"
```

LLM을 사용하지 않을 때도 `Cell 27`의 **규칙 기반 추천 근거/Gap/면접 질문** 생성이 동작하므로, 대시보드에 최소한의 설명이 표시됩니다.

LLM 해설 프롬프트는 세 가지로 분리되어 있습니다.

```text
Cell 28: 하드스킬 Evidence 시스템 프롬프트
          — JD 요구 스킬과 CV 원문을 비교해 V1(원문 인용) / V2(맥락 추론)로 판정

Cell 29: 소프트스킬 시스템 프롬프트
          — S1(협업) / S2(커뮤니케이션) / S3(문제해결) / S4(자기주도) / S5(적응력)
            각각 0/1로 판정하고 CV 원문 근거를 인용

Cell 30: JD-CV 매칭 해석 프롬프트
          — 추천 요약 / 강점 / Gap / 리스크 / 예상 면접 질문을 한 번에 생성
```

---

### 결과 저장

`Cell 40 ~ 46`에서 진행합니다.

매 실행마다 타임스탬프 파일을 새로 생성하지 않고, `latest_` 접두사가 붙은 고정 파일을 덮어씁니다. 실행 이력은 작은 index CSV(`run_index.csv`) 하나로만 관리합니다.

```python
# Cell 43 | 결과 저장
# 전체/JD별 결과를 latest 고정 파일에 갱신
OUTPUT_MATCHING_CSV_LATEST  = MATCHING_OUTPUT_DIR / "latest_matching_result_all.csv"
OUTPUT_MATCHING_XLSX_LATEST = MATCHING_OUTPUT_DIR / "latest_matching_result_all.xlsx"
```

---

## Output files

```text
data/
├── 08_matching/
│   ├── latest_matching_result_all.csv / .xlsx
│   │   — 모든 JD × Top 10 후보 결과
│   │
│   ├── latest_matching_result_855plus10_v5.csv / .xlsx
│   │   — 실제 CV 855명 + 가상 CV 10명 = 865명 전체를 final_score 기준 정렬
│   │
│   ├── latest_matching_result_855only_v5.csv / .xlsx
│   │   — 가상 CV를 제외한 실제 CV 855명만
│   │
│   ├── latest_matching_result_all_semantics.csv / .xlsx  ← v5 신규
│   │   — Gold/가상 CV 제외 + JD별 Top-N 제한 없이 전원 맥락 해석 컬럼 포함
│   │
│   └── by_jd/
│       └── jd_{id}/
│           └── latest_matching_result_jd_{id}.csv / .xlsx
│               — 특정 JD 1건의 결과만 분리
│
├── 09_dashboard/
│   ├── latest_dashboard_summary_all.csv / .xlsx
│   │   — 대시보드 연결용 fact table
│   │
│   └── by_jd/
│       └── jd_{id}/
│           └── latest_dashboard_summary_jd_{id}.csv / .xlsx
│
└── 10_evaluation/
    ├── evaluation_report_latest.xlsx
    │   — Recall / Precision / MRR / NDCG 계산 결과
    ├── weight_config_validation_latest.xlsx
    │   — 가중치 설정별 성능 비교
    └── grid_search_hybrid_v26.xlsx
        — Grid Search 결과 전체
```

주요 출력 컬럼 (`latest_matching_result_all`):

```text
jd_id, candidate_id, rank, final_score, final_score_100
skill_score, task_score, domain_score, role_score, ai_skill_overlap_score
retrieval_score_norm, keyword_baseline_score_norm, column_score
radar_skill, radar_task, radar_domain, radar_role, radar_experience, radar_softskill
match_reason_rule_based, gap_rule_based, interview_questions_rule_based
(LLM 활성화 시) hardskill_evidence_json, softskill_evidence_json,
                 llm_summary, llm_strength, llm_gap, llm_interview_questions
```

---

## Notes

- **CV 데이터 비공개**: `candidate_profile_jo_all_FINAL5.xlsx`와 매칭 결과물(`08_matching/`, `09_dashboard/`)은 개인정보 보호를 위해 이 저장소에 포함되지 않습니다. `.gitignore`에 `data/05_profiles/`와 출력 디렉터리가 등록되어 있는지 확인하세요.

- **API Key 보안**: `Cell 02`에 `GEMINI_API_KEY`를 직접 입력했다면 커밋 전에 반드시 삭제하세요. `.env` 파일을 사용하는 경우 `.gitignore`에 포함되어 있는지 확인하세요.

- **실행 환경 경로**: `Cell 01`의 `WINDOWS_PROJECT_ROOT = Path("D:/003.AIPJT")`는 원 개발 환경 기준값입니다. 해당 경로가 없는 환경에서는 `find_project_root()`가 자동으로 현재 작업 폴더 기준 탐색으로 전환하므로, 그대로 두어도 정상 동작합니다.

- **Cross-Encoder 속도**: CPU 환경에서 Cross-Encoder 재채점은 JD 수 × `TOP_N_CROSS_ENCODER`에 비례해 시간이 걸립니다. 처음 실행할 때는 `MANUAL_TARGET_JD_IDS`를 1~2개로, `TOP_N_CROSS_ENCODER`를 20~30으로 줄여 먼저 동작을 확인하는 것을 권장합니다.

- **TOP_N_RETRIEVAL 주의**: `TOP_N_RETRIEVAL = 865`(전체 후보 수)로 설정하면 1차 Retrieval의 필터링 효과가 없어집니다. 이 경우 Retrieval 단계는 점수 산출에만 기여하고, 후보 좁히기는 Cross-Encoder 이후 단계에서만 이루어집니다.

- **Pipeline 검증**: `Cell 44`에서 Taxonomy → Retrieval → 정밀 점수 → 가중치 → 대시보드 저장까지 실제 사용된 방법과 주요 컬럼 상태를 한 번에 점검할 수 있습니다.

---

## Roadmap

- [ ] 노트북을 CLI 실행 가능한 Python 스크립트(`.py`)로 전환
- [ ] 노트북 4분할: `07_embedding_build` / `08_retrieval_rerank` / `09_evaluation` / `10_dashboard`
- [ ] Ground Truth 정답 라벨 150~200건으로 확대
- [ ] 도메인 적응 임베딩 실험: 채용 도메인 fine-tuned 임베딩과 general BGE-M3 비교 (`ablation A5`)
- [ ] 컨설턴트 클릭 로그 피드백 루프 — 실사용 선택 데이터 기반 가중치 재튜닝
- [ ] 벡터 DB 도입 검토 (CV 규모 확장 시 ChromaDB 등)

---

## License

> TODO — 라이선스는 추후 결정 예정입니다. 라이선스가 확정되기 전까지는 별도 허가 없이 재배포·상업적 이용을 금지합니다.

---

## Contact

- GitHub: [@doctorSong1](https://github.com/doctorSong1)
- Repository: [doctorSong1/universe](https://github.com/doctorSong1/universe)

이슈나 개선 제안은 GitHub Issues로 남겨주세요.



