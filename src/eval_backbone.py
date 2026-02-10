import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, TRAIN_DIR
from src.model_factory import build_model


def get_class_names(train_dir: Path):
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


def load_and_prepare_image(image_path: str) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def true_label_from_path(image_path: str) -> str:
    return Path(image_path).parent.name


def sample_images(base_dir: str, n: int):
    base = Path(base_dir)
    all_imgs = []
    for class_dir in base.iterdir():
        if class_dir.is_dir():
            for p in class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    all_imgs.append(str(p))
    np.random.shuffle(all_imgs)
    return all_imgs[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    class_names = get_class_names(TRAIN_DIR)
    num_classes = len(class_names)

    model, _ = build_model(args.backbone, num_classes=num_classes)
    model.load_weights(args.weights)

    imgs = sample_images(args.data_dir, args.n)

    correct = 0
    for img_path in imgs:
        x = load_and_prepare_image(img_path)
        probs = model.predict(x, verbose=0)[0]
        pred = class_names[int(np.argmax(probs))]
        true = true_label_from_path(img_path)

        if pred == true:
            correct += 1
            print("✅", pred, "| true:", true)
        else:
            print("❌", pred, "| true:", true)

    acc = 100 * correct / len(imgs)
    print(f"\n📊 Quick accuracy on {len(imgs)} images: {acc:.2f}%")


if __name__ == "__main__":
    main()
