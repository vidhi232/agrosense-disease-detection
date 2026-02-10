import argparse
import tensorflow as tf
from src.data_pipeline import get_datasets
from src.model_factory import build_model
from src.config import IMG_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    print("\n✅ Loading datasets...")
    train_ds, valid_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print(f"✅ Classes: {num_classes}")

    print("\n✅ Building model...")
    model, base = build_model(args.backbone, num_classes)

    print("\n✅ Loading Stage-1 weights...")
    model.load_weights(args.weights)

    print("\n🔓 Unfreezing last 30 layers for fine-tuning...")
    base.trainable = True

    for layer in base.layers[:-30]:
        layer.trainable = False

    print("\n⚙ Re-compiling model for fine-tune...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"models/{args.backbone}_stage2.weights.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        save_weights_only=True,
        verbose=1,
    )

    print("\n🚀 Starting Stage-2 fine-tuning...")
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=[checkpoint],
    )

    print("\n🏁 Fine-tuning complete.")


if __name__ == "__main__":
    main()
