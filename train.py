import os
import sys
import shutil
import random
import multiprocessing
from pathlib import Path
from ultralytics import YOLO


def split_dataset(dataset_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Veri setini train/val olarak böler.
    Orijinal verilere dokunmaz, sembolik link veya kopyalama ile yeni yapı oluşturur.
    """
    images_src = os.path.join(dataset_dir, "images")
    labels_src = os.path.join(dataset_dir, "labels")

    # Çıktı klasörlerini oluştur
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Tüm görüntü dosyalarını listele
    image_files = sorted([
        f for f in os.listdir(images_src)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])

    # Tekrarlanabilir rastgele bölme
    random.seed(seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"  Train: {len(train_files)} görüntü")
    print(f"  Val:   {len(val_files)} görüntü")

    copied = {"train": 0, "val": 0}

    for split, files in [("train", train_files), ("val", val_files)]:
        for img_file in files:
            # Görüntüyü kopyala
            src_img = os.path.join(images_src, img_file)
            dst_img = os.path.join(output_dir, "images", split, img_file)

            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            # Etiket dosyasını kopyala
            label_name = Path(img_file).stem + ".txt"
            src_label = os.path.join(labels_src, label_name)
            dst_label = os.path.join(output_dir, "labels", split, label_name)

            if os.path.exists(src_label) and not os.path.exists(dst_label):
                # Segment verisi varsa temizle (sadece bbox formatı bırak)
                with open(src_label, 'r') as f:
                    lines = f.readlines()

                cleaned_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO bbox formatı: class x_center y_center width height
                        # 5'ten fazla değer varsa segment verisi var, sadece ilk 5'i al
                        cleaned_lines.append(" ".join(parts[:5]) + "\n")

                with open(dst_label, 'w') as f:
                    f.writelines(cleaned_lines)

                copied[split] += 1

    return len(train_files), len(val_files)


def create_yaml(output_dir, yaml_path):
    """Create data.yaml with train/val split configuration."""
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

nc: 1
names: ['tank']
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Orijinal veri seti konumu
    DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "tank_veriseti")
    if os.path.isdir(os.path.join(SCRIPT_DIR, "images")):
        DATASET_DIR = SCRIPT_DIR

    # Bölünmüş veri seti çıktı konumu
    SPLIT_DIR = os.path.join(SCRIPT_DIR, "dataset_split")
    YAML_PATH = os.path.join(SCRIPT_DIR, "data.yaml")

    # ─── ADIM 1: Veri Seti Kontrolü ─────────────────────────────────
    print("=" * 60)
    print("  TANK TESPİT MODELİ — OPTİMİZE EĞİTİM")
    print("=" * 60)

    print("\n[1/5] Veri seti kontrol ediliyor...")
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"\n[HATA] images veya labels klasörü bulunamadı!")
        print(f"  Aranan konum: {DATASET_DIR}")
        sys.exit(1)

    img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    lbl_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    print(f"  Görüntü: {img_count}, Etiket: {lbl_count}")

    # ─── ADIM 2: Train/Val Bölme ────────────────────────────────────
    print("\n[2/5] Veri seti train/val olarak bölünüyor (%80/%20)...")

    if os.path.isdir(SPLIT_DIR):
        # Daha önce bölünmüşse tekrar bölme
        existing_train = len(os.listdir(os.path.join(SPLIT_DIR, "images", "train"))) if os.path.isdir(os.path.join(SPLIT_DIR, "images", "train")) else 0
        if existing_train > 0:
            print(f"  Önceden bölünmüş veri seti bulundu ({existing_train} train görüntü). Atlanıyor.")
        else:
            split_dataset(DATASET_DIR, SPLIT_DIR)
    else:
        split_dataset(DATASET_DIR, SPLIT_DIR)

    # data.yaml oluştur
    create_yaml(SPLIT_DIR, YAML_PATH)
    print(f"  data.yaml güncellendi: train/val ayrı klasörler.")

    # ─── ADIM 3: Model Eğitimi ──────────────────────────────────────
    print("\n[3/5] RTX 4070 Laptop GPU ile eğitim başlatılıyor...")
    print("  Model: YOLOv8n (Raspberry Pi için optimize nano model)")
    print("  Hedef: Karanlık ortam + Düşük false positive\n")

    model = YOLO('yolov8n.pt')

    try:
        results = model.train(
            data=YAML_PATH,
            epochs=100,              # 50→100: Daha iyi yakınsama
            imgsz=640,               # Eğitimde yüksek çözünürlük (Pi'de 416'ya düşürülecek)
            batch=16,                # 8GB VRAM için ideal
            device=0,                # RTX 4070 Laptop GPU
            workers=2,               # Windows multiprocessing uyumlu
            cache='disk',            # RAM yetersiz, disk cache kullan
            amp=True,                # Mixed precision → hız artışı
            name="tank_kamikaze_v2",
            exist_ok=True,           # Aynı isimli klasörü yeniden kullan
            pretrained=True,         # COCO pretrained ağırlıklardan başla

            # ── Karanlık Ortam Augmentasyonları ──
            hsv_h=0.02,              # Renk tonu kaydırma (kamuflaj çeşitliliği)
            hsv_s=0.7,               # Doygunluk değişimi (gece→gündüz renk farkı)
            hsv_v=0.8,               # Parlaklık: %0-%160 arası → KARANLIK SİMÜLASYONU
            erasing=0.5,             # Rastgele bölge karartma (kısmi kapanma simülasyonu)

            # ── Geometrik Augmentasyonlar (Genelleme) ──
            degrees=15.0,            # ±15° rotasyon (eğik tanklar)
            translate=0.2,           # %20 konum kaydırma
            scale=0.6,               # %40-%160 ölçek değişimi (mesafe simülasyonu)
            fliplr=0.5,              # Yatay çevirme
            flipud=0.3,              # Dikey çevirme (drone perspektifi)
            mosaic=1.0,              # Mozaik: farklı arka planları karıştırır
            mixup=0.15,              # Görüntü karıştırma (aşırı ezber önleme)

            # ── False Positive Azaltma ──
            close_mosaic=10,         # Son 10 epochta mozaik KAPAT → gerçek görüntülerle ince ayar
            patience=25,             # 25 epoch iyileşme yoksa erken dur

            # ── Kayıp Fonksiyonu Ağırlıkları ──
            box=7.5,                 # Bounding box kaybı
            cls=1.0,                 # Sınıf kaybı (tek sınıf için artırıldı)
            dfl=1.5,                 # Distribution focal loss

            # ── Öğrenme Oranı ──
            lr0=0.01,                # Başlangıç LR
            lrf=0.01,                # Final LR (lr0 * lrf = 0.0001)
            warmup_epochs=5.0,       # 3→5 warmup (kararlı başlangıç)
            optimizer='AdamW',       # SGD yerine AdamW → küçük veri setlerinde daha kararlı
        )

        # ─── ADIM 4: Model Değerlendirme ────────────────────────────
        print("\n[4/5] Model değerlendiriliyor...")
        save_dir = results.save_dir if hasattr(results, 'save_dir') else os.path.join("runs", "detect", "tank_kamikaze_v2")
        best_pt = os.path.join(save_dir, "weights", "best.pt")

        if os.path.exists(best_pt):
            best_model = YOLO(best_pt)
            metrics = best_model.val(data=YAML_PATH, device=0)
            print(f"\n  ╔══════════════════════════════════════╗")
            print(f"  ║  DOĞRULAMA SONUÇLARI                 ║")
            print(f"  ╠══════════════════════════════════════╣")
            print(f"  ║  mAP@50:      {metrics.box.map50:.4f}               ║")
            print(f"  ║  mAP@50-95:   {metrics.box.map:.4f}               ║")
            print(f"  ║  Precision:   {metrics.box.mp:.4f}               ║")
            print(f"  ║  Recall:      {metrics.box.mr:.4f}               ║")
            print(f"  ╚══════════════════════════════════════╝")
        else:
            print(f"  [UYARI] best.pt bulunamadı: {best_pt}")
            best_pt = os.path.join(save_dir, "weights", "last.pt")

        # ─── ADIM 5: Raspberry Pi Export ─────────────────────────────
        print("\n[5/5] Raspberry Pi için model dışa aktarılıyor...")

        export_model = YOLO(best_pt)

        # ONNX Export (FP16, 416px — Pi için optimize)
        print("\n  → ONNX formatı (genel uyumluluk)...")
        export_model.export(
            format="onnx",
            imgsz=416,               # 640→416: Pi'de %58 daha az hesaplama
            simplify=True,           # ONNX graph optimizasyonu
            half=True,               # FP16: %50 bellek tasarrufu
        )

        # NCNN Export (ARM optimizasyonlu — Pi için en iyi performans)
        print("  → NCNN formatı (Raspberry Pi ARM optimize)...")
        try:
            export_model.export(
                format="ncnn",
                imgsz=416,
                half=True,
            )
            print("  [OK] NCNN modeli oluşturuldu!")
        except Exception as e:
            print(f"  [UYARI] NCNN export başarısız (opsiyonel): {e}")
            print("  ONNX modeli Pi'de de kullanılabilir.")

        print(f"\n{'=' * 60}")
        print(f"  ✅ EĞİTİM TAMAMLANDI!")
        print(f"  📁 Model konumu: {best_pt}")
        print(f"  🍓 Pi ONNX: {best_pt.replace('.pt', '.onnx')}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n[HATA] Eğitim sırasında bir sorun oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()