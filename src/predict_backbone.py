import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np

from src.model_factory import build_model
from src.data_pipeline import get_datasets
from src.config import IMG_SIZE


def load_image(image_path: str) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    # class names come from training folder order
    _, _, class_names = get_datasets()
    num_classes = len(class_names)

    model, _ = build_model(args.backbone, num_classes)

    w2 = Path(f"models/{args.backbone}_stage2.weights.h5")
    w1 = Path(f"models/{args.backbone}_stage1.weights.h5")

    if w2.exists():
        model.load_weights(str(w2))
        used = str(w2)
    else:
        model.load_weights(str(w1))
        used = str(w1)

    x = load_image(args.image)
    probs = model.predict(x, verbose=0)[0]

    topk = min(args.topk, len(class_names))
    idxs = np.argsort(probs)[::-1][:topk]

    print(f"\n🖼️ Image: {args.image}")
    print(f"✅ Weights used: {used}\n")
    print(f"Top {topk} predictions:")
    for i, idx in enumerate(idxs, start=1):
        print(f"{i}) {class_names[idx]} — {probs[idx]*100:.2f}%")


if __name__ == "__main__":
    main()

