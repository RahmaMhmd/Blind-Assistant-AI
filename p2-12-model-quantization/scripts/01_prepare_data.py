import os
from datasets import load_dataset
from tqdm import tqdm

# إعدادات الفولدر وعدد الصور (المطلوب من 100 لـ 300 صورة)
OUTPUT_DIR = "data/calibration"
NUM_IMAGES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"جاري سحب {NUM_IMAGES} صورة حقيقية من COCO للمعايرة (Calibration)...")

# استخدام streaming=True لتحميل الصور المطلوبة فقط دون تنزيل الداتاسيت بالكامل
dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)

count = 0
for item in tqdm(dataset, total=NUM_IMAGES):
    if count >= NUM_IMAGES:
        break
    
    img = item["image"]
    
    # تحويل الصورة لـ RGB لتجنب أي خطأ في الأبعاد لو في صور أبيض وأسود
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    img_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.jpg")
    img.save(img_path, format="JPEG", quality=95)
    count += 1

print(f"\nتم بنجاح! حفظنا {count} صورة في الفولدر: {OUTPUT_DIR}")