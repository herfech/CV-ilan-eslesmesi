#!/usr/bin/env python3
"""
push.py — CV Matching projesini GitHub'a otomatik yükler.

"""

import subprocess
import sys
import datetime

# ── Ayarlar ───────────────────────────────────────────────────
REPO_URL   = "https://github.com/herfech/CV-ilan-eslesmesi.git"
BRANCH     = "main"

# ── Yardımcı fonksiyon ────────────────────────────────────────
def calistir(komut: str, hata_durdurusun: bool = True) -> str:
    """Komutu çalıştırır, çıktıyı döndürür."""
    sonuc = subprocess.run(
        komut, shell=True,
        capture_output=True, text=True, encoding="utf-8"
    )
    if sonuc.stdout.strip():
        print(f"  {sonuc.stdout.strip()}")
    if sonuc.returncode != 0:
        if sonuc.stderr.strip():
            print(f"  ⚠️  {sonuc.stderr.strip()}")
        if hata_durdurusun:
            print(f"\n❌ Komut başarısız: {komut}")
            sys.exit(1)
    return sonuc.stdout.strip()

def separator():
    print("─" * 50)

# ── Ana akış ──────────────────────────────────────────────────
def main():
    tarih = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Commit mesajı: argümandan veya varsayılan
    if len(sys.argv) > 1:
        mesaj = " ".join(sys.argv[1:])
    else:
        mesaj = f"Güncelleme — {tarih}"

    print()
    print("🚀 GitHub'a yükleme başlıyor...")
    separator()

    # 1. Remote bağlantı kontrolü / ekleme
    print("🔗 Remote kontrol ediliyor...")
    remote_list = calistir("git remote", hata_durdurusun=False)
    if "origin" not in remote_list:
        print("  Remote bulunamadı, ekleniyor...")
        calistir(f'git remote add origin {REPO_URL}')
        print("  ✅ Remote eklendi.")
    else:
        # URL'i güncelle (değişmiş olabilir)
        calistir(f'git remote set-url origin {REPO_URL}')
        print("  ✅ Remote hazır.")

    separator()

    # 2. Git init (eğer henüz yapılmamışsa)
    import os
    if not os.path.exists(".git"):
        print("🔧 Git başlatılıyor...")
        calistir("git init")
        calistir(f"git branch -M {BRANCH}")
        print("  ✅ Git başlatıldı.")
        separator()

    # 3. Durum kontrolü
    print("📋 Değişen dosyalar:")
    durum = calistir("git status --short", hata_durdurusun=False)
    if not durum:
        print("  ℹ️  Yüklenecek değişiklik yok. Proje zaten güncel.")
        print()
        return

    separator()

    # 4. Tüm dosyaları ekle
    print("➕ Dosyalar ekleniyor...")
    calistir("git add .")
    print("  ✅ Tüm dosyalar eklendi.")

    separator()

    # 5. Commit
    print(f"📝 Commit oluşturuluyor: '{mesaj}'")
    calistir(f'git commit -m "{mesaj}"')
    print("  ✅ Commit oluşturuldu.")

    separator()

    # 6. Branch ayarla
    calistir(f"git branch -M {BRANCH}", hata_durdurusun=False)

    # 7. Push
    print(f"⬆️  GitHub'a gönderiliyor ({BRANCH})...")
    calistir(f"git push -u origin {BRANCH}")

    separator()
    print(f"✅ Yükleme tamamlandı!")
    print(f"🌐 Repo: https://github.com/herfech/CV-ilan-eslesmesi")
    print()

if __name__ == "__main__":
    main()
