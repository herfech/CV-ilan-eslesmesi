# 🎯 CV – İlan Eşleşmesi

> **İnsan Kaynakları · NLP · Kosinüs Benzerliği · Streamlit**

Aday CV'lerinin iş ilanlarıyla metinsel uyumunu ölçen, TF-IDF vektörleştirme ve Kosinüs Benzerliği algoritması kullanan bir İnsan Kaynakları otomasyon sistemi.

---

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Dosya Yapısı](#dosya-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Kullanım Kılavuzu](#kullanım-kılavuzu)
- [Matematiksel Temel](#matematiksel-temel)
- [Teknoloji Yığını](#teknoloji-yığını)

---

## Proje Hakkında

Bu proje, İnsan Kaynakları süreçlerini otomatikleştirmek amacıyla geliştirilmiştir. Sistem şu adımları gerçekleştirir:

1. CV ve iş ilanı metinlerini temizler ve normalleştirir
2. TF-IDF algoritmasıyla metinleri sayısal vektörlere dönüştürür
3. Kosinüs Benzerliği formülüyle uyum skorunu hesaplar
4. En yüksek skora sahip Top-N adayı sıralar
5. Sonuçları Streamlit dashboard üzerinden görselleştirir

---

## Özellikler

- 🔤 Türkçe + İngilizce metin desteği
- 📊 TF-IDF ve Sentence-Transformers vektörleştirme seçeneği
- 📎 `.txt` dosyası yükleme (birden fazla aday)
- ✏️ Manuel CV metin girişi
- 📈 Plotly ile interaktif grafikler (bar + gauge)
- ⬇️ Sonuçları CSV olarak indirme
- 🎨 Karanlık tema Streamlit arayüzü
- 🐳 Docker desteği

---

## Dosya Yapısı

```
CV_MATCHING/
├── app.py                           # Streamlit ana uygulaması
├── matching_engine.py               # NLP motoru
├── sample_data.py                   # Örnek veriler
├── test_matching.py                 # Terminal testi
├── requirements.txt                 # Bağımlılıklar
├── Dockerfile                       # Docker imajı
├── docker-compose.yml               # Docker Compose
├── notebooks/
│   ├── 01_eda_ve_onisleme.ipynb     # Keşifsel analiz
│   └── 02_model_karsilastirma.ipynb # Model karşılaştırması
├── rapor/
│   └── cv_ilan_eslesme_rapor.tex    # LaTeX teknik raporu
└── cvs_ornek/
    ├── aday1_cv.txt
    └── aday2_cv.txt
```

### Her Dosya Ne Yapar?

| Dosya | Açıklama |
|---|---|
| `app.py` | Streamlit arayüzü — tarayıcıda açılan dashboard |
| `matching_engine.py` | Metin ön işleme, TF-IDF, Cosine Similarity mantığı |
| `sample_data.py` | Uygulamada hazır gelen 5 örnek CV ve 1 iş ilanı |
| `test_matching.py` | Streamlit olmadan terminalde hızlı test |
| `requirements.txt` | `pip install -r requirements.txt` ile kurulur |
| `Dockerfile` | Uygulamayı Docker ile çalıştırmak için |
| `docker-compose.yml` | Docker Compose ile tek komutla başlatmak için |
| `notebooks/01_*.ipynb` | EDA, ön işleme adımları, TF-IDF heatmap |
| `notebooks/02_*.ipynb` | TF-IDF ile Sentence-Transformers karşılaştırması |
| `rapor/*.tex` | Overleaf'te derlenen Türkçe teknik rapor |
| `cvs_ornek/*.txt` | Dosya yükleme özelliğini test etmek için örnek CV'ler |

---

## Kurulum

### Gereksinimler

- Python 3.9+
- pip

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADIN/cv-matching.git
cd cv-matching

# 2. Sanal ortam oluştur
python -m venv venv

# Linux / Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## Çalıştırma

### Streamlit Dashboard (Önerilen)

```bash
streamlit run app.py
```

Tarayıcıda otomatik açılır: `http://localhost:8501`

### Terminal Testi

```bash
python test_matching.py
```

### Jupyter Notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
```

### Docker ile

```bash
# İmaj oluştur ve çalıştır
docker build -t cv-matching .
docker run -p 8501:8501 cv-matching

# Ya da Docker Compose ile
docker compose up
```

---

## Kullanım Kılavuzu

1. **Sol panel** → İş ilanı metnini yapıştır
2. **Sağ panel** → CV kaynağını seç:
   - `📂 Örnek CV'leri kullan` → 5 hazır aday yüklenir
   - `📎 .txt dosyası yükle` → Kendi dosyalarını yükle
   - `✏️ Metni elle gir` → Doğrudan yapıştır
3. **"🚀 Analizi Başlat"** butonuna bas
4. Gauge grafiği, bar grafik ve tabloyu incele
5. **CSV indir** ile sonuçları dışa aktar

---

## Matematiksel Temel

```
TF-IDF(t, d) = TF(t,d) × log(N / df(t))

Kosinüs Benzerliği:
similarity = cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

| Skor | Değerlendirme |
|---|---|
| 0.60 – 1.00 | 🟢 Yüksek uyum |
| 0.35 – 0.60 | 🟡 Orta uyum |
| 0.00 – 0.35 | 🔴 Düşük uyum |

---

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Arayüz | Streamlit |
| NLP | Scikit-learn (TF-IDF) |
| Görselleştirme | Plotly |
| Metin İşleme | NLTK |
| Veri | Pandas, NumPy |
| Konteyner | Docker |
| Derin NLP (opsiyonel) | Sentence-Transformers |