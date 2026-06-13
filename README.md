# 🎯 CV – İlan Eşleşmesi (Proje [50])

**Ders:** Doğal Dil İşleme
**Öğretim Elemanı:** Dr. Rabia Yaşa Koştaş
**Final Ödev-2** | Teslim: 15 Haziran 2026

**Proje Ekibi:**
1. Mario Enrique Motede Dasilva | 2411081503
2. Heriberto Fernandez Chale | 2411081509
3. Matias Fernando Ndong Owono Obiang | 2311081634

---

## 📖 Proje Açıklaması

Bu proje, **aday CV'leri ile iş ilanlarının metinsel uyumunu** ölçen bir NLP sistemidir. İki aşamadan oluşur:

1. **TF-IDF tabanlı eşleştirme motoru** (`app.py`, `matching_engine.py`) — Streamlit dashboard üzerinden gerçek zamanlı CV–İlan eşleştirmesi
2. **Word2Vec tabanlı akademik analiz** (`Final_Odev2.ipynb`) — 280 belgelik veri seti üzerinde 16 Word2Vec modeli eğitilip karşılaştırmalı değerlendirme yapılır (Final Ödev-2)

---

## 📁 Proje Dosya Yapısı

```
cv_matching/
├── app.py                      → Streamlit dashboard (ana uygulama)
├── matching_engine.py          → TF-IDF + Cosine Similarity NLP motoru
├── sample_data.py              → 5 örnek CV ve 1 örnek iş ilanı
├── create_dataset.py           → 280 belgelik veri seti oluşturucu
├── test_matching.py            → Terminalde hızlı test scripti
├── push.py                     → GitHub'a otomatik yükleme betiği
├── requirements.txt            → Python bağımlılıkları
├── .gitignore                  → Git tarafından yok sayılan dosyalar
│
├── Final_Odev2.ipynb            → ⭐ ÖDEV-2 ana notebook (Word2Vec, 16 model)
│
├── data/
│   ├── cv_jobs_raw.csv          → Ham veri seti (280 belge: 250 CV + 30 ilan)
│   ├── cv_jobs_dataset.csv      → document_id | content formatında veri
│   ├── lemmatized.csv           → Lemmatization uygulanmış veri
│   ├── stemmed.csv               → Stemming uygulanmış veri
│   ├── cosine_evaluation.csv     → 16 model için cosine skor tablosu
│   ├── semantic_evaluation.csv   → 16 model için anlamsal puan tablosu
│   └── jaccard_matrix.csv         → 16x16 Jaccard benzerlik matrisi
│
├── models/
│   └── word2vec_*.model          → 16 eğitilmiş Word2Vec modeli
│
│
├── notebooks/plots
│   ├── 01_eda_ve_onisleme.ipynb  → Keşifsel veri analizi + 6 adım ön işleme
│   └── 02_model_karsilastirma.ipynb → TF-IDF vs Word2Vec karşılaştırması
│
├── cvs_örnek/
│   └── *.txt                     → Örnek yüklenebilir CV dosyaları
│
└── rapor/
    ├── cv_ilan_eslesme_rapor.tex → LaTeX raporu (Overleaf)
    ├── CV_Ilan_Eslesmesi_Rapor.pdf  → PDF raporu (DBS'e yüklenecek)
    └── GU_LOGO.png                 → Üniversite logosu
```

---

## ⚙️ Kurulum

### 1. Gereksinimler

- **Python 3.9 veya üzeri**
- pip (Python paket yöneticisi)

### 2. Sanal Ortam Oluşturma

```bash
# Sanal ortam oluştur
python -m venv venv

# Aktif et (Windows)
venv\Scripts\activate

# Aktif et (Mac/Linux)
source venv/bin/activate
```

### 3. Bağımlılıkları Kurma

```bash
pip install -r requirements.txt
```

`requirements.txt` içeriği ve her birinin amacı:

| Paket | Amaç |
|---|---|
| `streamlit` | Web dashboard arayüzü |
| `scikit-learn` | TF-IDF vektörleştirme + Cosine Similarity |
| `pandas` | Veri okuma/işleme (CSV) |
| `numpy` | Vektör işlemleri |
| `plotly` | İnteraktif grafikler (dashboard) |
| `nltk` | Stopword listesi, tokenization, lemmatization, stemming |
| `gensim` | Word2Vec model eğitimi (Final Ödev-2) |
| `pdfplumber` | PDF dosyalarından metin çıkarma |
| `matplotlib`, `seaborn` | Notebook grafikleri (Final_Odev2, 01, 02) |
| `jupyter`, `ipykernel` | Jupyter Notebook çalıştırma |

### 4. NLTK Verilerini İndirme (otomatik)

Notebook'lar ve `matching_engine.py` ilk çalıştırıldığında gerekli NLTK verilerini (`punkt`, `stopwords`, `wordnet`) otomatik indirir. Manuel indirmek isterseniz:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 🚀 Çalıştırma

### A) Streamlit Dashboard (Ana Uygulama)

```bash
streamlit run app.py
```

Tarayıcıda otomatik açılır: `http://localhost:8501`

**Dashboard özellikleri:**
- İş ilanını metin olarak yapıştırma **veya** `.txt`/`.pdf` dosyası yükleme
- CV kaynağı seçimi: örnek CV'ler / proje veri seti (280 belge) / dosya yükleme / manuel giriş
- Vektörleştirme yöntemi seçimi: **TF-IDF**, **Sentence-Transformers** veya **Word2Vec** (Final Ödev-2'nin 16 modelinden biri)
- Sonuç görselleştirme: gauge chart, bar chart, sıralama tablosu, CSV export
- Aday CV detaylarının ön işlenmiş halini görüntüleme

### B) Terminal Testi (Streamlit'siz Hızlı Test)

```bash
python test_matching.py
```

5 örnek CV ile 1 örnek iş ilanını TF-IDF + Cosine Similarity ile karşılaştırır, sonucu ASCII bar grafik olarak terminalde gösterir.

### C) Veri Setini Yeniden Oluşturma

```bash
python create_dataset.py
```

`data/cv_jobs_raw.csv` ve `data/cv_jobs_dataset.csv` dosyalarını oluşturur: 250 CV (10 kategori) + 30 iş ilanı (10 kategori x 3 seviye) = **280 belge**.

> ⚠️ Bu komutu çalıştırmak mevcut veri setinin üzerine yazar. `Final_Odev2.ipynb` bu dosyalara bağımlıdır.

### D) Final Ödev-2 — Word2Vec Analizi (⭐ Ana Teslim)

Jupyter ortamında (VS Code, Jupyter Lab, vb.) açın:

```
Final_Odev2.ipynb
```

Tüm hücreleri **sırasıyla** (Run All veya tek tek) çalıştırın. Notebook şunları yapar:

1. 280 belgelik veri setini yükler
2. 6 adımlı ön işleme uygular → `lemmatized.csv`, `stemmed.csv` oluşturur
3. Gensim ile **16 Word2Vec modeli** eğitir (CBOW/Skip-gram x window 2/4 x dim 100/300, lemmatized+stemmed) → `models/`
4. Örnek giriş metni (cv_001) için her modelde **Top-5 benzer belge** bulur
5. **3 değerlendirme** yapar:
   - Cosine Benzerlik Tablosu → `data/cosine_evaluation.csv`
   - Anlamsal Değerlendirme (1-5 puan) → `data/semantic_evaluation.csv`
   - Jaccard 16x16 Matrisi + Heatmap → `data/jaccard_matrix.csv`, `plots/04_jaccard_heatmap.png`
6. Sonuçları yorumlar ve model önerileri sunar

**Süre:** Model eğitimi (hücre 7) yaklaşık 2-5 dakika sürer.

### E) Ek Notebook'lar (Destekleyici Analiz)

```
notebooks/01_eda_ve_onisleme.ipynb     → Keşifsel veri analizi + 6 adım ön işleme detaylı gösterim
notebooks/02_model_karsilastirma.ipynb → TF-IDF vs Word2Vec karşılaştırması, n-gram optimizasyonu
```

Bu notebook'lar `Final_Odev2.ipynb`'nin ürettiği `models/` klasöründeki modellere bağımlıdır — önce Final_Odev2.ipynb çalıştırılmalıdır.

---

## 🧠 Teknik Detaylar

### Metin Ön İşleme (6 Adım)

```
Ham Metin -> Unicode Normalizasyonu -> Kucuk Harf -> URL/Email Kaldirma
          -> Sayi Kaldirma -> Noktalama Kaldirma -> Stopword Eleme
          -> (Lemmatization veya Stemming)
```

### Kosinüs Benzerliği

```
cos(theta) = (A . B) / (||A|| x ||B||)
```

- Sonuç aralığı: [0, 1] — 1'e yakın = yüksek uyum
- TF-IDF için: belge bazında TF-IDF vektörleri arası benzerlik
- Word2Vec için: belgedeki kelime vektörlerinin ortalaması (Zero Vector koruması ile)

### Word2Vec — 16 Model Parametreleri

| Veri Seti | Model Tipi | Window | Vector Size |
|---|---|---|---|
| lemmatized / stemmed | CBOW / Skip-gram | 2 / 4 | 100 / 300 |

2 (veri seti) x 2 (model tipi) x 2 (window) x 2 (boyut) = **16 model**

İsimlendirme: `word2vec_{lemmatized|stemmed}_{cbow|skipgram}_win{2|4}_dim{100|300}.model`

---

## 🔧 Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `FileNotFoundError: data/cv_jobs_raw.csv` | `python create_dataset.py` çalıştırın |
| `NameError` (notebook'ta) | **Restart Kernel** yapıp tüm hücreleri sırayla çalıştırın (Run All) |
| `ModuleNotFoundError: gensim` | `pip install gensim` |
| PDF yüklenemiyor | `pip install pdfplumber` |
| `plots/` klasörü bulunamadı | `mkdir plots` (proje kök dizininde) |
| NLTK verisi hatası | Yukarıdaki NLTK indirme komutunu çalıştırın |

---

## 📤 GitHub'a Yükleme

```bash
python push.py "commit mesajınız"
```

**Repo:** https://github.com/herfech/CV-ilan-eslesmesi

---

## 📄 Rapor

Final Ödev-2 raporu `rapor/` klasöründe 3 formatta mevcuttur:

- `CV_Ilan_Eslesmesi_Rapor.pdf` → **DBS'e yüklenecek dosya**
- `cv_ilan_eslesme_rapor.tex` → Overleaf'te derlemek için (GU_LOGO.png ile birlikte yükleyin)

Rapor 4 bölümden oluşur: **Giriş**, **Yöntem**, **Sonuçlar ve Değerlendirme**, **Sonuç ve Öneriler**.
