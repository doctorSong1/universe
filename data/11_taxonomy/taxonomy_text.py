from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DEFAULT_STOPWORDS = {
    "및",
    "등",
    "관련",
    "업무",
    "수행",
    "경험",
    "가능",
    "필수",
    "우대",
    "보유",
    "있는",
    "위한",
    "활용",
    "기반",
    "진행",
    "관리",
    "운영",
    "개발",
    "구축",
    "분석",
    "설계",
    "프로젝트",
    "서비스",
    "시스템",
    "데이터",
    "모델",
    "사용",
    "대한",
    "통한",
    "또는",
    "주요",
    "주요직무",
    "담당",
    "담당업무",
    "자격",
    "자격요건",
    "우대사항",
    "직무",
    "역량",
    "기술",
    "능력",
    "실무",
    "협업",
    "지원",
    "개선",
    "고도화",
    "최적화",
    "자동화",
    "처리",
    "생성",
    "적용",
    "제공",
    "기획",
    "제안",
    "도출",
    "수립",
    "작성",
    "포함",
    "이상",
    "이하",
    "경력",
    "신입",
    "학력",
    "무관",
    "전공",
    "전형",
    "서류",
    "면접",
    "합격",
    "팀원",
    "사원",
    "대리",
    "과장",
    "차장",
    "부장",
    "임원",
    "with",
    "and",
    "or",
    "the",
    "for",
    "to",
    "of",
    "in",
    "on",
    "a",
    "an",
}


DEFAULT_STOPWORD_PATTERNS = (
    re.compile(r"^\d+년(차)?$"),
    re.compile(r"^\d+개월$"),
    re.compile(r"^\d+$"),
)


def clean_text(text: object) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_simple_stopword_yaml(path: Path) -> dict[str, list[str]]:
    config: dict[str, list[str]] = {}
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            config[section] = []
            continue
        if section and line.strip().startswith("- "):
            value = line.strip()[2:].strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            config[section].append(value)

    return config


def load_stopwords(path: str | Path | None = None) -> tuple[set[str], list[re.Pattern[str]]]:
    if path is None:
        return set(DEFAULT_STOPWORDS), list(DEFAULT_STOPWORD_PATTERNS)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stopword file not found: {path}")

    config = _load_simple_stopword_yaml(path)
    words: set[str] = set()

    for key in ("korean", "english"):
        words.update(clean_text(word).lower() for word in config.get(key, []) if clean_text(word))

    patterns = [re.compile(pattern) for pattern in config.get("patterns", [])]
    return words, patterns


def is_stopword(term: str, stopwords: set[str], patterns: Iterable[re.Pattern[str]]) -> bool:
    term = clean_text(term)
    if not term:
        return True
    if term in stopwords or term.lower() in stopwords:
        return True
    return any(pattern.search(term) for pattern in patterns)


KOREAN_PARTICLE_PATTERN = re.compile(
    r"(으로서|으로써|으로|에서|에게|부터|까지|보다|처럼|만큼|이고|이며|하고|"
    r"은|는|이|가|을|를|와|과|로|도|만|에|의)$"
)

KOREAN_VERB_ENDING_PATTERN = re.compile(
    r"(합니다|했습니다|됩니다|됩니다|있는|있음|있고|있으며|있으신|"
    r"경험이|경험을|경험은)$"
)


def normalize_token(token: str) -> str:
    token = clean_text(token).lower()
    token = re.sub(r"^[\-–—\*\d\.\)\(]+", "", token).strip()
    token = re.sub(r"[\.,;:!\?\)\]\}]+$", "", token).strip()

    if re.fullmatch(r"[가-힣]{3,}", token):
        token = KOREAN_PARTICLE_PATTERN.sub("", token)
        token = KOREAN_VERB_ENDING_PATTERN.sub("", token)

    return token.strip()


@lru_cache(maxsize=1)
def _load_morph_backend():
    try:
        from kiwipiepy import Kiwi

        return "kiwi", Kiwi()
    except Exception:
        pass

    try:
        from konlpy.tag import Okt

        return "okt", Okt()
    except Exception:
        pass

    return None, None


def tokenize_taxonomy_terms(
    text: object,
    stopwords: set[str] | None = None,
    stopword_patterns: Iterable[re.Pattern[str]] | None = None,
    use_morph: bool = True,
    min_len: int = 2,
) -> list[str]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    stopword_patterns = stopword_patterns or DEFAULT_STOPWORD_PATTERNS
    text = clean_text(text)
    tokens: list[str] = []

    backend_name, backend = _load_morph_backend() if use_morph else (None, None)

    if backend_name == "kiwi":
        keep_tags = {"NNG", "NNP", "SL", "SN"}
        tokens = [token.form for token in backend.tokenize(text) if token.tag in keep_tags]
    elif backend_name == "okt":
        tokens = backend.nouns(text)
        tokens += re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]*", text)
    else:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]*|[가-힣]{2,}", text)

    normalized = []
    for token in tokens:
        token = normalize_token(token)
        if (
            len(token) >= min_len
            and not is_stopword(token, stopwords, stopword_patterns)
        ):
            normalized.append(token)

    return normalized


def split_skill_terms(text: object) -> list[str]:
    text = clean_text(text)
    parts = re.split(r"[,/|·•\n;]+", text)
    terms = []
    for part in parts:
        part = normalize_token(part)
        if len(part) >= 2:
            terms.append(part)
    return terms


def extract_ngrams(
    text: object,
    stopwords: set[str] | None = None,
    stopword_patterns: Iterable[re.Pattern[str]] | None = None,
    ngram_range: tuple[int, int] = (1, 3),
    min_len: int = 2,
) -> list[str]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    stopword_patterns = stopword_patterns or DEFAULT_STOPWORD_PATTERNS
    tokens = tokenize_taxonomy_terms(
        text,
        stopwords=stopwords,
        stopword_patterns=stopword_patterns,
        use_morph=True,
        min_len=min_len,
    )

    grams = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            if (
                len(gram) >= min_len
                and not is_stopword(gram, stopwords, stopword_patterns)
            ):
                grams.append(gram)
    return grams


def count_terms(
    texts: Iterable[object],
    mode: str = "ngram",
    ngram_range: tuple[int, int] = (1, 3),
    stopwords: set[str] | None = None,
    stopword_patterns: Iterable[re.Pattern[str]] | None = None,
) -> Counter[str]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    stopword_patterns = stopword_patterns or DEFAULT_STOPWORD_PATTERNS
    counter: Counter[str] = Counter()

    for text in texts:
        if mode == "skill":
            terms = split_skill_terms(text)
            terms += extract_ngrams(text, stopwords, stopword_patterns, ngram_range=(1, 1))
        else:
            terms = extract_ngrams(text, stopwords, stopword_patterns, ngram_range=ngram_range)

        for term in terms:
            term = clean_text(term)
            if not is_stopword(term, stopwords, stopword_patterns):
                counter[term] += 1

    return counter
