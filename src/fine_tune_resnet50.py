import tensorflow as tf
from pathlib import Path

from src.data_pipeline import get_datasets
from src.config import IMG_SIZE


ROOT_DIR = Path(__file__).resolve().parent.parent
STAGE1_WEIGHTS = ROOT_DIR / "models/resnet50_stage1.weights.h5"
STAGE2_WEIGHTS = ROOT_DIR / "models/resnet50_stage2.weights.h5"

EPOCHS_STAGE2 = 2  # keep small for Mac CPU


def build_resnet50_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(inputs)
    x = tf.keras.applications.resnet.preprocess_input(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def main():
    print("\n✅ Loading datasets...")
    train_ds, valid_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print(f"✅ Classes: {num_classes}")

    print("\n✅ Building model...")
    model, base = build_resnet50_model(num_classes)

    print("\n✅ Loading Stage 1 weights...")
    model.load_weights(STAGE1_WEIGHTS)

    # ---- Fine-tuning: unfreeze last N layers of base ----
    print("\n✅ Unfreezing last 30 layers of ResNet50 base...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    # IMPORTANT: recompile after changing trainable flags
    print("\n✅ Compiling (small learning rate for fine-tuning)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    print("\n✅ Starting Stage 2 fine-tuning...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS_STAGE2,
    )

    print("\n✅ Saving Stage 2 weights...")
    Path("models").mkdir(exist_ok=True)
    model.save_weights(STAGE2_WEIGHTS)
    print(f"✅ Saved: {STAGE2_WEIGHTS}")


if __name__ == "__main__":
    main()
