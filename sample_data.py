"""
sample_data.py
──────────────
Sistemi hemen test etmek için 1 iş ilanı ve 5 örnek mini-CV.

Senaryo: Bir fintech şirketi kıdemli Python/ML geliştiricisi arıyor.
Adaylar farklı arka plan ve yetkinliklere sahip; bu sayede
benzerlik skorlarında belirgin farklılıklar gözlemlenebilir.
"""

# ── İş İlanı ─────────────────────────────────────────────────
JOB_DESCRIPTION = """
Kıdemli Makine Öğrenmesi Mühendisi – Fintech

Şirket: FinAI Technologies | İstanbul (Hibrit)

Sorumluluklar:
- Büyük veri setleri üzerinde makine öğrenmesi ve derin öğrenme modelleri geliştirmek
- Python ile üretim ortamına hazır ML pipeline'ları tasarlamak ve dağıtmak
- Scikit-learn, TensorFlow, PyTorch gibi kütüphaneleri kullanarak model eğitimi ve optimizasyonu
- NLP tekniklerini (metin sınıflandırma, entity extraction, embedding) finans alanına uygulamak
- Docker ve Kubernetes ile MLOps süreçlerini yönetmek
- SQL ve NoSQL veritabanlarından veri çekme ve analiz
- RESTful API'lar aracılığıyla modelleri servis etmek (FastAPI)
- Takım içi kod incelemeleri ve teknik mentorluk

Aranan Nitelikler:
- Python'da en az 4 yıl deneyim
- Makine öğrenmesi algoritmalarına hakimiyet (regression, classification, clustering, NLP)
- Büyük veri araçları: Spark, Hadoop, ya da Kafka deneyimi tercih sebebi
- Git, CI/CD, ve Agile metodolojilere aşinalık
- İngilizce teknik belgeler okuyabilmek
- Fintech veya bankacılık sektöründe çalışmış olmak artı

Yan Haklar:
- Rekabetçi maaş + hisse opsiyonu
- Sağlık sigortası
- Eğitim bütçesi (yıllık 3.000 USD)
- Uzaktan çalışma esnekliği
"""

# ── 5 Örnek Aday CV'si ────────────────────────────────────────
SAMPLE_CVS = [
    (
        "Ahmet Yılmaz",
        """
        Ahmet Yılmaz | Senior ML Engineer | ahmet@example.com | İstanbul

        Özet:
        6 yıllık Python ve makine öğrenmesi deneyimine sahip kıdemli mühendis.
        Fintech alanında 3 yıl çalışarak kredi risk modelleri ve dolandırıcılık tespiti
        sistemleri geliştirdim. TensorFlow, PyTorch ve Scikit-learn konularında uzmanlık.

        Deneyim:
        - PayTech A.Ş. (2021-2024): Kıdemli ML Mühendisi
          * Kredi skorlama modeli (XGBoost) geliştirdim; doğruluk oranı %94'e çıktı
          * NLP ile müşteri şikayetlerini otomatik sınıflandıran sistem kurdum
          * FastAPI + Docker ile model deployment pipeline'ı oluşturdum
          * Kafka ile gerçek zamanlı dolandırıcılık tespiti altyapısı kurdum

        - DataLab (2018-2021): ML Mühendisi
          * Scikit-learn ile müşteri churn tahmin modeli (AUC: 0.91)
          * Apache Spark ile büyük veri analizi
          * Agile/Scrum ortamında çalışma

        Teknik Beceriler:
        Python, TensorFlow, PyTorch, Scikit-learn, FastAPI, Docker, Kubernetes,
        Kafka, Apache Spark, SQL, MongoDB, Git, CI/CD (GitHub Actions)

        Eğitim:
        Boğaziçi Üniversitesi – Bilgisayar Mühendisliği Yüksek Lisans (2018)

        Diller: Türkçe (ana dil), İngilizce (C1 – IELTS 7.5)
        """,
    ),
    (
        "Zeynep Kaya",
        """
        Zeynep Kaya | Data Scientist | zeynep@example.com | Ankara

        Hakkımda:
        4 yıllık veri bilimi deneyimi. İstatistik ve makine öğrenmesi odaklı çalışmalar.
        Python, R ve SQL konularında güçlüyüm. NLP projelerinde metin sınıflandırma
        ve duygu analizi üzerine çalıştım. Akademik geçmişim güçlü.

        İş Deneyimi:
        - TechCorp (2022-2024): Data Scientist
          * Python ve Scikit-learn ile müşteri segmentasyon modelleri
          * BERT tabanlı duygu analizi modeli (Twitter verisi)
          * SQL sorguları ve PostgreSQL yönetimi
          * Jupyter Notebook, Pandas, Matplotlib ile veri keşfi

        - Araştırma Asistanı – ODTÜ (2020-2022)
          * Doğal dil işleme araştırmaları
          * PyTorch ile transformer modelleri

        Beceriler:
        Python, R, SQL, Scikit-learn, PyTorch, HuggingFace Transformers,
        Pandas, NumPy, Matplotlib, Seaborn, Git, Jupyter

        Eğitim:
        ODTÜ – İstatistik Lisans, Yapay Zeka Yüksek Lisans (devam ediyor)

        Diller: Türkçe, İngilizce (B2)
        Sertifikalar: Google ML Crash Course, Kaggle Expert
        """,
    ),
    (
        "Mert Demir",
        """
        Mert Demir | Backend Developer | mert@example.com | İzmir

        Ben kimim:
        5 yıllık backend geliştirme deneyimi. Ağırlıklı olarak Python web framework'leri
        (Django, FastAPI, Flask) üzerinde çalıştım. Veritabanı tasarımı ve API geliştirme
        konularında deneyimliyim. Makine öğrenmesine ilgim var, henüz profesyonel
        deneyimim kısıtlı.

        Projeler ve Deneyim:
        - FinServ Ltd (2022-2024): Python Backend Developer
          * FastAPI ile RESTful API geliştirme (60+ endpoint)
          * PostgreSQL ve Redis ile veritabanı optimizasyonu
          * Docker ve Docker Compose ile konteynerizasyon
          * Birkaç küçük ML modeli (LinearRegression, RandomForest) entegrasyonu

        - Startup XYZ (2019-2022): Full-stack Developer
          * Django + React ile web uygulamaları
          * AWS (EC2, S3, Lambda) ile cloud deployment

        Teknik Yetkinlikler:
        Python (5 yıl), Django, FastAPI, Flask, PostgreSQL, Redis, Docker,
        Git, AWS, JavaScript, React

        Eğitim:
        Ege Üniversitesi – Yazılım Mühendisliği (2019)

        Not: ML alanında kendimi geliştirmeye devam ediyorum.
        Coursera – Machine Learning Specialization (tamamlandı, 2023)
        """,
    ),
    (
        "Elif Şahin",
        """
        Elif Şahin | UI/UX Designer & Frontend Developer | elif@example.com

        Yaratıcı problem çözücü. 6 yıllık kullanıcı deneyimi tasarımı ve frontend
        geliştirme. Figma, Adobe XD, HTML/CSS/JavaScript alanlarında uzmanlık.
        Kullanıcı araştırması ve prototipleme konusunda güçlüyüm.

        Kariyer Özeti:
        - DesignHub (2021-2024): Lead UX Designer
          * Mobil ve web uygulamaları için UI/UX tasarımları
          * Figma ile interaktif prototipleme
          * A/B testleri ve kullanıcı araştırması
          * React ile bazı frontend geliştirmeleri

        - Ajans (2018-2021): Junior Designer
          * Grafik tasarım ve web sitesi geliştirme
          * HTML, CSS, JavaScript, WordPress

        Beceriler:
        Figma, Adobe XD, Sketch, HTML5, CSS3, JavaScript, React (temel),
        Kullanıcı araştırması, Wireframing, Prototipleme, Accessibility

        Eğitim:
        İstanbul Teknik Üniversitesi – Endüstriyel Tasarım

        Diller: Türkçe, İngilizce (B2), Almanca (A2)

        Portfolio: elif-design.portfolio.com
        """,
    ),
    (
        "Can Arslan",
        """
        Can Arslan | Junior Data Analyst | can@example.com | Bursa

        Üniversiteden yeni mezun, veri analitiği alanında kariyer yapmak istiyorum.
        Python ve SQL öğrendim. Kaggle yarışmalarına katılıyorum. Öğrenmeye açık,
        hevesli ve çalışkan biriyim.

        Eğitim:
        Uludağ Üniversitesi – Endüstri Mühendisliği (2024)
        GPA: 3.4 / 4.0

        Staj:
        - Yıldız Holding (2023, 3 ay): Data Analyst Stajyeri
          * Excel ve Power BI ile raporlama
          * Python (Pandas) ile veri temizleme
          * SQL sorguları

        Kişisel Projeler:
        - Kaggle: Titanic survival prediction (Accuracy: 0.82)
        - GitHub: Hava durumu veri analizi (Matplotlib + Seaborn)
        - Web scraping scripti (BeautifulSoup + Requests)

        Teknik Bilgiler:
        Python (1.5 yıl – öğrenci düzeyi), SQL (temel), Excel (ileri),
        Power BI, Pandas, NumPy, Matplotlib

        Sertifikalar:
        - Google Data Analytics Professional Certificate
        - Python for Everybody – Coursera

        Diller: Türkçe, İngilizce (B1)

        Hedef: ML alanında kendimi geliştirerek veri bilimcisi olmak.
        """,
    ),
]
