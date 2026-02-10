import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, TRAIN_DIR


def get_class_names(train_dir: Path):
    # Folder names = class names
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


def build_resnet50_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # our loaded image will be scaled to [0,1]
    x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(inputs)
    x = tf.keras.applications.resnet.preprocess_input(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def load_and_prepare_image(image_path: str) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


def get_true_label_from_path(image_path: str) -> str:
    # .../train/ClassName/file.jpg OR .../valid/ClassName/file.jpg
    return Path(image_path).parent.name


def predict_one(image_path: str, weights_path: str, top_k: int = 3):
    class_names = get_class_names(TRAIN_DIR)
    num_classes = len(class_names)

    model = build_resnet50_model(num_classes)
    model.load_weights(weights_path)

    x = load_and_prepare_image(image_path)
    probs = model.predict(x, verbose=0)[0]  # (38,)

    top_indices = probs.argsort()[-top_k:][::-1]

    true_label = get_true_label_from_path(image_path)
    pred_label = class_names[top_indices[0]]
    pred_conf = probs[top_indices[0]] * 100

    print("\n🖼️ Image:", image_path)
    print("✅ True label:", true_label)
    print(f"🤖 Predicted: {pred_label} ({pred_conf:.2f}%)")

    if pred_label == true_label:
        print("✅ Result: CORRECT")
    else:
        print("❌ Result: WRONG")

    print("\nTop {} predictions:".format(top_k))
    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank}) {class_names[idx]} — {probs[idx]*100:.2f}%")


def sample_images_from_dataset(base_dir: str, n: int = 10) -> list[str]:
    base = Path(base_dir)
    all_images = []
    for class_dir in base.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    all_images.append(str(img_path))
    if len(all_images) == 0:
        return []
    np.random.shuffle(all_images)
    return all_images[:n]


def quick_test(base_dir: str, weights_path: str, n: int = 10):
    class_names = get_class_names(TRAIN_DIR)
    num_classes = len(class_names)

    model = build_resnet50_model(num_classes)
    model.load_weights(weights_path)

    images = sample_images_from_dataset(base_dir, n=n)
    if not images:
        print("❌ No images found for quick test.")
        return

    correct = 0
    for img_path in images:
        true_label = get_true_label_from_path(img_path)
        x = load_and_prepare_image(img_path)
        probs = model.predict(x, verbose=0)[0]
        pred_label = class_names[int(np.argmax(probs))]
        is_correct = (pred_label == true_label)
        correct += int(is_correct)
        print(("✅" if is_correct else "❌"), pred_label, "| true:", true_label, "|", Path(img_path).name)

    acc = (correct / len(images)) * 100
    print(f"\n📊 Quick test accuracy on {len(images)} random images: {acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--weights", default="models/resnet50_stage1.weights.h5", help="Path to weights file")
    parser.add_argument("--topk", type=int, default=3, help="Top K predictions to show")
    parser.add_argument("--quick_test_dir", help="Folder like valid/ClassName... (e.g., the whole valid folder)")
    parser.add_argument("--n", type=int, default=10, help="Number of images for quick test")
    args = parser.parse_args()

    if args.quick_test_dir:
        quick_test(args.quick_test_dir, args.weights, n=args.n)
    elif args.image:
        predict_one(args.image, args.weights, top_k=args.topk)
    else:
        print("❌ Provide either --image or --quick_test_dir")
