import os
from pathlib import Path

DATASET_ROOT = Path("../data/raw/New Plant Diseases Dataset(Augmented)")


print("\n📂 DATASET ROOT:", DATASET_ROOT.resolve())

# Find train/valid folders anywhere inside this root
train_dirs = list(DATASET_ROOT.rglob("train"))
valid_dirs = list(DATASET_ROOT.rglob("valid"))

print("\n🔎 Searching for train/valid folders...")

if not train_dirs:
    raise FileNotFoundError("❌ Could not find a 'train' folder inside the dataset root.")
if not valid_dirs:
    raise FileNotFoundError("❌ Could not find a 'valid' folder inside the dataset root.")

train_path = train_dirs[0]
valid_path = valid_dirs[0]

print("✅ Train folder found:", train_path)
print("✅ Valid folder found:", valid_path)

# Count classes
classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
print("\n🌿 Total classes:", len(classes))
print("\n📌 First 10 class names:")
for c in classes[:10]:
    print(" -", c)

# Count total images in train and valid
def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)

train_images = count_images(train_path)
valid_images = count_images(valid_path)

print("\n🖼 Total training images:", train_images)
print("🖼 Total validation images:", valid_images)
print("\n🎉 Dataset check complete!\n")
