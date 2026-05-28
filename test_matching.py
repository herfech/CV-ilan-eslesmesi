"""
test_matching.py
────────────────
Streamlit olmadan, doğrudan terminalde çalışan test scripti.
Sonuçları tablo ve ASCII bar grafik olarak gösterir.

Çalıştırma:
    python test_matching.py
"""

from matching_engine import MatchingEngine
from sample_data import JOB_DESCRIPTION, SAMPLE_CVS


def ascii_bar(score: float, width: int = 30) -> str:
    """Terminalde basit yatay bar grafik."""
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def score_label(score: float) -> str:
    pct = score * 100
    if pct >= 60:
        return "🟢 YÜKSEK"
    elif pct >= 35:
        return "🟡 ORTA"
    else:
        return "🔴 DÜŞÜK"


def main():
    print("\n" + "═" * 60)
    print("  🎯  CV – İLAN EŞLEŞMESİ | HR Matching Engine Test")
    print("═" * 60)

    candidates = dict(SAMPLE_CVS)

    print(f"\n📋 İş İlanı (ilk 200 karakter):")
    print(f"   {JOB_DESCRIPTION.strip()[:200]}...")

    print(f"\n👥 Toplam aday: {len(candidates)}")
    print("─" * 60)

    # Motor oluştur ve çalıştır
    engine  = MatchingEngine(use_sentence_transformers=False)
    results = engine.rank_candidates(
        job_description=JOB_DESCRIPTION,
        candidates=candidates,
        top_n=5,
    )

    print("\n🏅 SONUÇLAR – Top 5 Aday\n")
    print(f"  {'Sıra':<5} {'Aday Adı':<20} {'Skor':>8}  {'Bar Grafik':<32}  {'Değerlendirme'}")
    print("  " + "─" * 78)

    for r in results:
        bar = ascii_bar(r["score"])
        print(
            f"  #{r['rank']:<4} {r['candidate']:<20} {r['score']*100:>6.2f}%  "
            f"{bar}  {score_label(r['score'])}"
        )

    best = results[0]
    print("\n" + "─" * 60)
    print(f"  🏆 En İyi Eşleşme  : {best['candidate']}")
    print(f"  ⭐ Uyum Skoru       : %{best['score']*100:.2f}")
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  📊 Ortalama Skor   : %{avg*100:.2f}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
