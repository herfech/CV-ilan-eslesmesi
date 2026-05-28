"""
╔══════════════════════════════════════════════════════════════╗
║         CV - İlan Eşleşmesi | HR Matching Dashboard          ║
║         Geliştirici: Senior Data Scientist & Full-stack      ║
╚══════════════════════════════════════════════════════════════╝

Açıklama:
    Bu uygulama, aday CV'lerini iş ilanlarıyla karşılaştırarak
    Kosinüs Benzerliği (Cosine Similarity) skoru hesaplar ve
    en uygun 5 adayı sıralar.

Kullanılan Teknolojiler:
    - Streamlit  → Web arayüzü
    - Scikit-learn (TF-IDF + Cosine Similarity) → NLP & Benzerlik
    - Plotly     → İnteraktif grafikler
    - NLTK       → Metin ön işleme
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matching_engine import MatchingEngine, preprocess_text
from sample_data import JOB_DESCRIPTION, SAMPLE_CVS

# ── Sayfa Konfigürasyonu ──────────────────────────────────────
st.set_page_config(
    page_title="CV-İlan Eşleşmesi | HR Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Özel CSS Stilleri ─────────────────────────────────────────
st.markdown("""
<style>
    /* Ana arka plan */
    .stApp { background-color: #0f1117; }

    /* Başlık kartı */
    .hero-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.2);
    }

    /* Metrik kartı */
    .metric-card {
        background: linear-gradient(145deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: #e94560; transform: translateY(-2px); }

    /* Skor rozeti */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .score-high   { background: #0d6b3e; color: #4ade80; }
    .score-medium { background: #7c4b00; color: #fbbf24; }
    .score-low    { background: #7c1d1d; color: #f87171; }

    /* Sidebar stili */
    .css-1d391kg { background-color: #0d1117; }

    /* Tablo stili */
    .dataframe { border-radius: 8px !important; }

    /* Bilgi kutusu */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }

    /* Adım kartı */
    .step-card {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────
def score_badge(score: float) -> str:
    """Skora göre renk rozeti HTML oluşturur."""
    pct = score * 100
    if pct >= 60:
        css = "score-high"
        label = f"🟢 {pct:.1f}%"
    elif pct >= 35:
        css = "score-medium"
        label = f"🟡 {pct:.1f}%"
    else:
        css = "score-low"
        label = f"🔴 {pct:.1f}%"
    return f'<span class="score-badge {css}">{label}</span>'


def build_bar_chart(results: list[dict]) -> go.Figure:
    """Plotly ile yatay bar grafik oluşturur."""
    names  = [r["candidate"] for r in results]
    scores = [round(r["score"] * 100, 2) for r in results]
    colors = ["#e94560" if s >= 60 else "#f59e0b" if s >= 35 else "#6366f1"
              for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="white", size=13),
        hovertemplate="<b>%{y}</b><br>Uyum Skoru: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="📊 Aday Uyum Skoru Karşılaştırması", font=dict(color="white", size=16)),
        xaxis=dict(title="Uyum Skoru (%)", range=[0, 110], gridcolor="#2d3561",
                   tickfont=dict(color="#a0aec0"), title_font=dict(color="#a0aec0")),
        yaxis=dict(gridcolor="#2d3561", tickfont=dict(color="white")),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#1a1f2e",
        height=max(350, len(results) * 60 + 100),
        margin=dict(l=20, r=80, t=60, b=40),
        font=dict(family="monospace"),
        showlegend=False,
    )

    # Referans çizgisi – %50 eşiği
    fig.add_vline(x=50, line_dash="dash", line_color="#4ade80",
                  annotation_text="Eşik: 50%", annotation_font_color="#4ade80",
                  annotation_position="top right")
    return fig


def build_gauge(score: float, name: str) -> go.Figure:
    """En iyi aday için gauge chart oluşturur."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score * 100, 1),
        delta={"reference": 50, "valueformat": ".1f"},
        title={"text": f"🏆 En İyi Aday<br><b>{name}</b>", "font": {"size": 14, "color": "white"}},
        number={"suffix": "%", "font": {"color": "#e94560", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#a0aec0"},
            "bar": {"color": "#e94560"},
            "bgcolor": "#1a1f2e",
            "bordercolor": "#2d3561",
            "steps": [
                {"range": [0, 35], "color": "#2d1f2e"},
                {"range": [35, 60], "color": "#2d2a1f"},
                {"range": [60, 100], "color": "#1f2d27"},
            ],
            "threshold": {"line": {"color": "#4ade80", "width": 3}, "value": 60},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#1a1f2e", font_color="white",
        height=260, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    st.divider()

    vectorizer_choice = st.selectbox(
        "Vektörleştirme Yöntemi",
        ["TF-IDF (Hızlı)", "Sentence-Transformers (Derin)"],
        help="TF-IDF daha hızlı; Sentence-Transformers anlam bazlı daha derin analiz yapar.",
    )
    use_sentence_transformers = "Sentence" in vectorizer_choice

    top_n = st.slider("Gösterilecek Aday Sayısı (Top-N)", 1, 10, 5)

    st.divider()
    st.markdown("""
    <div class="step-card">
    <b>📐 Cosine Similarity</b><br><br>
    <code>sim(A,B) = (A·B) / (‖A‖ × ‖B‖)</code><br><br>
    <small>1'e yakın → Yüksek uyum<br>0'a yakın → Düşük uyum</small>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <small style='color:#6b7280'>
    ℹ️ Streamlit · Scikit-learn<br>
    TF-IDF · Cosine Similarity<br>
    Plotly · NLTK
    </small>
    """, unsafe_allow_html=True)


# ── Ana İçerik ────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1 style='color:#e94560; margin:0; font-size:2rem;'>🎯 CV – İlan Eşleşmesi</h1>
    <p style='color:#a0aec0; margin:0.5rem 0 0 0; font-size:1rem;'>
        İnsan Kaynakları · Metin Benzerliği · Kosinüs Skoru · NLP
    </p>
</div>
""", unsafe_allow_html=True)

# ── Girdi Bölümü ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📋 İş İlanı")
    job_text = st.text_area(
        "İş tanımını buraya yapıştırın:",
        value=JOB_DESCRIPTION,
        height=260,
        placeholder="İş ilanı metnini buraya yapıştırın...",
    )

with col_right:
    st.markdown("### 👤 CV'ler")

    # ── CV Kaynağı: kullanıcı üç seçenekten birini seçer ──────────
    cv_source = st.radio(
        "CV kaynağını seçin:",
        ["📂 Örnek CV'leri kullan", "📎 .txt dosyası yükle", "✏️ Metni elle gir"],
        horizontal=True,
    )

    candidate_texts = {}   # her zaman boş başlar, seçime göre dolar

    # ── SEÇENEK 1: Örnek CV'ler ───────────────────────────────────
    if cv_source == "📂 Örnek CV'leri kullan":
        candidate_texts = dict(SAMPLE_CVS)
        st.success(f"✅ {len(candidate_texts)} örnek aday CV'si yüklendi:")
        for isim in candidate_texts:
            st.markdown(f"&nbsp;&nbsp;&nbsp;• {isim}")

    # ── SEÇENEK 2: .txt dosyası yükle ────────────────────────────
    elif cv_source == "📎 .txt dosyası yükle":
        st.info(
            "📌 **Beklenen format:** Her aday için ayrı bir .txt dosyası. "
            "Dosya adı (uzantısız) aday adı olarak kullanılır."
        )
        uploaded_files = st.file_uploader(
            "Bir veya birden fazla .txt dosyası seçin",
            type=["txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            for uf in uploaded_files:
                content = uf.read().decode("utf-8", errors="ignore").strip()
                if content:
                    isim = uf.name.replace(".txt", "").replace("_", " ").replace("-", " ")
                    candidate_texts[isim] = content

            st.success(f"✅ {len(candidate_texts)} dosya başarıyla yüklendi:")
            for isim, metin in candidate_texts.items():
                kelime = len(metin.split())
                st.markdown(f"&nbsp;&nbsp;&nbsp;• **{isim}** — {kelime} kelime")
        else:
            st.warning("⬆️ Devam etmek için en az bir .txt dosyası yükleyin.")

    # ── SEÇENEK 3: Metni elle gir ────────────────────────────────
    elif cv_source == "✏️ Metni elle gir":
        n_manual = st.number_input("Kaç aday eklenecek?", 1, 20, 2, key="n_manual")
        for i in range(int(n_manual)):
            cols = st.columns([1, 2])
            with cols[0]:
                cname = st.text_input(f"Aday {i+1} Adı", value=f"Aday_{i+1}", key=f"cname_{i}")
            with cols[1]:
                ctext = st.text_area(f"{cname} CV Metni", height=80, key=f"ctext_{i}",
                                     placeholder="CV metnini buraya yapıştırın...")
            if cname and ctext.strip():
                candidate_texts[cname] = ctext

        if candidate_texts:
            st.success(f"✅ {len(candidate_texts)} aday hazır.")
        else:
            st.warning("✏️ En az bir aday adı ve CV metni girin.")

    # ── Analiz öncesi özet ────────────────────────────────────────
    if candidate_texts:
        st.markdown(
            f"<div style='margin-top:0.8rem;padding:0.6rem 1rem;"
            f"background:#1a2e1a;border-radius:8px;border:1px solid #3a6b3a;"
            f"color:#7ec87e;font-size:13px'>"
            f"🎯 <b>{len(candidate_texts)} aday</b> analiz için hazır</div>",
            unsafe_allow_html=True,
        )


# ── Analiz Butonu ─────────────────────────────────────────────
st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("🚀 Analizi Başlat", type="primary", use_container_width=True)

# ── Sonuçlar ──────────────────────────────────────────────────
if run_btn:
    if not job_text.strip():
        st.error("❌ Lütfen bir iş ilanı metni girin.")
    elif len(candidate_texts) == 0:
        st.error("❌ Lütfen en az bir aday CV'si ekleyin.")
    else:
        with st.spinner("🔄 Metinler vektörleştiriliyor ve benzerlik hesaplanıyor..."):
            engine = MatchingEngine(use_sentence_transformers=use_sentence_transformers)
            results = engine.rank_candidates(
                job_description=job_text,
                candidates=candidate_texts,
                top_n=min(top_n, len(candidate_texts)),
            )

        st.success(f"✅ Analiz tamamlandı! {len(candidate_texts)} aday arasından Top-{len(results)} listelendi.")

        # ── Özet Metrikleri ──────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        best = results[0]
        avg_score = sum(r["score"] for r in results) / len(results)
        above_50 = sum(1 for r in results if r["score"] >= 0.5)

        with m1:
            st.metric("🏆 En İyi Aday", best["candidate"])
        with m2:
            st.metric("⭐ En Yüksek Skor", f"{best['score']*100:.1f}%")
        with m3:
            st.metric("📊 Ortalama Skor", f"{avg_score*100:.1f}%")
        with m4:
            st.metric("✅ %50+ Uyum", f"{above_50} aday")

        st.markdown("---")

        # ── Gauge + Bar Grafik ────────────────────────────────
        g_col, b_col = st.columns([1, 2], gap="large")
        with g_col:
            st.plotly_chart(build_gauge(best["score"], best["candidate"]),
                            use_container_width=True)
        with b_col:
            st.plotly_chart(build_bar_chart(results), use_container_width=True)

        st.markdown("---")

        # ── Sıralama Tablosu ──────────────────────────────────
        st.markdown("### 🏅 Aday Sıralama Tablosu")

        df = pd.DataFrame(results)
        df.index += 1
        df["score_pct"] = df["score"].apply(lambda x: round(x * 100, 2))
        df["Uyum"] = df["score"].apply(
            lambda x: "🟢 Yüksek" if x >= 0.6 else ("🟡 Orta" if x >= 0.35 else "🔴 Düşük")
        )
        df_display = df[["candidate", "score_pct", "Uyum"]].rename(
            columns={"candidate": "Aday Adı", "score_pct": "Benzerlik Skoru (%)"}
        )
        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "Benzerlik Skoru (%)": st.column_config.ProgressColumn(
                    "Benzerlik Skoru (%)",
                    min_value=0, max_value=100, format="%.2f%%"
                )
            },
        )

        # ── CSV İndir ─────────────────────────────────────────
        csv = df_display.to_csv(index=True).encode("utf-8")
        st.download_button(
            "⬇️ Sonuçları CSV olarak indir",
            data=csv,
            file_name="eslesme_sonuclari.csv",
            mime="text/csv",
        )

        # ── Aday Detayları ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Aday CV Detayları (Ön İşlenmiş)")
        for r in results:
            with st.expander(f"📄 {r['candidate']} — Skor: {r['score']*100:.1f}%"):
                raw_text = candidate_texts[r["candidate"]]
                processed = preprocess_text(raw_text)
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    st.markdown("**Orijinal Metin:**")
                    st.text(raw_text[:500] + ("..." if len(raw_text) > 500 else ""))
                with tcol2:
                    st.markdown("**Ön İşlenmiş Metin (Token'lar):**")
                    st.text(processed[:500] + ("..." if len(processed) > 500 else ""))
