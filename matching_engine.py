"""
matching_engine.py
──────────────────
NLP çekirdeği: Metin ön işleme, vektörleştirme ve benzerlik hesaplama.

Desteklenen Yöntemler:
    1. TF-IDF + Cosine Similarity (Scikit-learn)
    2. Sentence-Transformers (opsiyonel, derin anlam tabanlı)

Matematiksel Temel:
    cos(θ) = (A · B) / (‖A‖ × ‖B‖)
    - A ve B: belge vektörleri
    - Sonuç: [0, 1] aralığında, 1 = tam uyum
"""

import re
import string
import unicodedata
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK stopword desteği (opsiyonel – eksikse basit liste kullanılır)
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt",     quiet=True)
    from nltk.corpus import stopwords as nltk_stopwords
    STOPWORDS_EN = set(nltk_stopwords.words("english"))
    STOPWORDS_TR = set(nltk_stopwords.words("turkish"))
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    # Temel İngilizce stopword listesi
    STOPWORDS_EN = {
        "i","me","my","myself","we","our","ours","ourselves","you","your",
        "yours","he","him","his","she","her","hers","it","its","they","them",
        "their","what","which","who","whom","this","that","these","those","am",
        "is","are","was","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","shall",
        "can","a","an","the","and","but","or","nor","for","yet","so","in",
        "on","at","to","from","of","with","by","about","as","into","through",
        "during","before","after","above","below","up","down","out","off",
        "over","under","again","then","once","here","there","when","where",
        "why","how","all","both","each","few","more","most","other","some",
        "such","no","not","only","own","same","than","too","very","just",
        "because","if","while","although","though","since","until","unless",
    }
    STOPWORDS_TR = {
        "bir","ve","ile","de","da","bu","o","şu","ben","sen","biz","siz",
        "için","olan","olan","olarak","gibi","kadar","sonra","önce","üzere",
        "ancak","fakat","ama","veya","ya","ne","ki","mi","mu","mü","mı",
        "her","hiç","çok","az","daha","en","hem","ise","değil","var","yok",
    }
    STOPWORDS_ALL = STOPWORDS_EN | STOPWORDS_TR


# ── Metin Ön İşleme ───────────────────────────────────────────
def preprocess_text(text: str, language: str = "mixed") -> str:
    """
    Ham metni temizler ve normalleştirir.

    Adımlar:
        1. Unicode normalizasyonu (NFD → NFKD)
        2. Küçük harfe çevirme
        3. URL ve e-posta kaldırma
        4. Sayı kaldırma (isteğe bağlı değiştirilebilir)
        5. Noktalama kaldırma
        6. Fazla boşluk temizleme
        7. Stopword eleme

    Args:
        text: Ham metin
        language: "en" | "tr" | "mixed"

    Returns:
        Temizlenmiş, token'lara ayrılmış metin
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unicode normalizasyonu
    text = unicodedata.normalize("NFKD", text)

    # 2. Küçük harf
    text = text.lower()

    # 3. URL ve e-posta kaldır
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 4. Sayıları kaldır (telefon, yıl vb.)
    text = re.sub(r"\b\d+\b", " ", text)

    # 5. Noktalama işaretlerini kaldır
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # 6. Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Stopword eleme
    if NLTK_AVAILABLE:
        stop = set()
        if language in ("en", "mixed"):
            stop |= STOPWORDS_EN
        if language in ("tr", "mixed"):
            stop |= STOPWORDS_TR
    else:
        stop = STOPWORDS_EN | STOPWORDS_TR

    tokens = [w for w in text.split() if w not in stop and len(w) > 2]
    return " ".join(tokens)


# ── Vektörleştirme Motoru ────────────────────────────────────
class TFIDFVectorizer:
    """TF-IDF tabanlı vektörleştirici (Scikit-learn)."""

    def __init__(self, ngram_range: tuple = (1, 2), max_features: int = 5000):
        self.model = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,          # TF'de log ölçeği
            analyzer="word",
        )
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.model.fit_transform(texts)
        self._fitted = True
        return matrix.toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit_transform() çağırın.")
        return self.model.transform(texts).toarray()


class SentenceTransformerVectorizer:
    """
    Sentence-Transformers tabanlı anlam vektörleştirici.
    Kurulum: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._available = True
        except ImportError:
            print("⚠️  sentence-transformers yüklü değil. TF-IDF'e geçiliyor.")
            self._available = False
            self._fallback = TFIDFVectorizer()

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._available:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self._fallback.fit_transform(texts)


# ── Benzerlik Hesaplama ───────────────────────────────────────
def cosine_similarity_manual(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Kosinüs benzerliğini sıfırdan hesaplar (eğitim / referans amaçlı).

    Formül:
        cos(θ) = (A · B) / (‖A‖ × ‖B‖)

    Args:
        vec_a: Birinci vektör (1D)
        vec_b: İkinci vektör (1D)

    Returns:
        Benzerlik skoru [0.0, 1.0]
    """
    dot_product = np.dot(vec_a, vec_b)           # A · B
    norm_a = np.linalg.norm(vec_a)               # ‖A‖
    norm_b = np.linalg.norm(vec_b)               # ‖B‖

    if norm_a == 0 or norm_b == 0:
        return 0.0                               # Sıfır vektör koruması

    return float(dot_product / (norm_a * norm_b))


# ── Ana Eşleştirme Motoru ────────────────────────────────────
class MatchingEngine:
    """
    CV – İlan eşleştirme motoru.

    Kullanım:
        engine = MatchingEngine()
        results = engine.rank_candidates(job_desc, candidates_dict, top_n=5)
    """

    def __init__(self, use_sentence_transformers: bool = False):
        self.use_st = use_sentence_transformers
        if use_sentence_transformers:
            self.vectorizer = SentenceTransformerVectorizer()
        else:
            self.vectorizer = None  # TF-IDF: her sorguda yeniden fit edilir

    def _vectorize_tfidf(self, job_text: str, cv_texts: list[str]) -> tuple:
        """TF-IDF ile vektör matrisi döndürür."""
        all_texts = [job_text] + cv_texts
        preprocessed = [preprocess_text(t) for t in all_texts]

        vec = TFIDFVectorizer(ngram_range=(1, 2))
        matrix = vec.fit_transform(preprocessed)

        job_vec = matrix[0]        # İlk satır = iş ilanı
        cv_vecs = matrix[1:]       # Geri kalanlar = CV'ler
        return job_vec, cv_vecs

    def _vectorize_st(self, job_text: str, cv_texts: list[str]) -> tuple:
        """Sentence-Transformers ile vektör matrisi döndürür."""
        all_texts = [job_text] + cv_texts
        matrix = self.vectorizer.encode(all_texts)
        return matrix[0], matrix[1:]

    def rank_candidates(
        self,
        job_description: str,
        candidates: dict[str, str],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Adayları iş ilanıyla karşılaştırarak sıralar.

        Args:
            job_description: İş ilanı metni
            candidates: {aday_adı: cv_metni} sözlüğü
            top_n: Döndürülecek en iyi aday sayısı

        Returns:
            [{"rank": int, "candidate": str, "score": float}, ...]
        """
        names    = list(candidates.keys())
        cv_texts = list(candidates.values())

        # Vektörleştir
        if self.use_st and hasattr(self.vectorizer, "_available") and self.vectorizer._available:
            job_vec, cv_vecs = self._vectorize_st(job_description, cv_texts)
        else:
            job_vec, cv_vecs = self._vectorize_tfidf(job_description, cv_texts)

        # Kosinüs benzerliği hesapla (Scikit-learn – hızlı)
        scores = cosine_similarity([job_vec], cv_vecs)[0]

        # Sıralama
        ranked_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append({
                "rank":      rank,
                "candidate": names[idx],
                "score":     float(scores[idx]),
            })

        return results
