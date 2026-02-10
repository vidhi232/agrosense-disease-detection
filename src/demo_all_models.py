import random
from pathlib import Path
import tensorflow as tf
import numpy as np

from src.model_factory import build_model
from src.data_pipeline import get_datasets
from src.config import IMG_SIZE

VALID_DIR = Path("data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid")

MODELS = ["resnet50", "mobilenetv2", "efficientnetb0", "densenet201"]


def load_img(path: Path):
    img = tf.keras.utils.load_img(str(path), target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    _, _, class_names = get_datasets()
    num_classes = len(class_names)

    all_images = []
    for ext in ["*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png"]:
        all_images.extend(list(VALID_DIR.glob(f"*/*{ext[1:]}")))

    n = 100
    n = min(n, len(all_images))  # ✅ prevents sample > population
    sample = random.sample(all_images, n)
    print(f"✅ Using {n} images for quick comparison (available: {len(all_images)})")

    results = {}

    for backbone in MODELS:
        model, _ = build_model(backbone, num_classes)

        w2 = Path(f"models/{backbone}_stage2.weights.h5")
        w1 = Path(f"models/{backbone}_stage1.weights.h5")
        model.load_weights(str(w2 if w2.exists() else w1))

        correct = 0
        for p in sample:
            true = p.parent.name
            pred = class_names[int(np.argmax(model.predict(load_img(p), verbose=0)[0]))]
            correct += int(pred == true)

        acc = 100.0 * correct / len(sample)
        results[backbone] = acc
        print(f"{backbone}: {acc:.2f}%")

    best = max(results, key=results.get)
    print(f"\n🏆 BEST (on this quick sample): {best.upper()}")


if __name__ == "__main__":
    main()
