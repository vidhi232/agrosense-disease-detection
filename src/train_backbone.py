import argparse
from pathlib import Path
import tensorflow as tf

from src.data_pipeline import get_datasets
from src.model_factory import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="resnet50 | densenet201 | efficientnetb0 | mobilenetv2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    backbone = args.backbone.lower()
    epochs = args.epochs
    lr = args.lr

    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("models/logs").mkdir(parents=True, exist_ok=True)

    print("\n✅ Loading datasets...")
    train_ds, valid_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print(f"✅ Classes: {num_classes}")

    print(f"\n✅ Building model: {backbone}")
    model, base = build_model(backbone, num_classes=num_classes)

    print("\n✅ Compiling...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
        ]
    )

    # Logs + safe checkpoints (weights only)
    csv_logger = tf.keras.callbacks.CSVLogger(f"models/logs/{backbone}_stage1_log.csv", append=False)

    ckpt_last = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"models/checkpoints/{backbone}_stage1_last.weights.h5",
        save_weights_only=True,
        save_best_only=False,
        verbose=1
    )

    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"models/checkpoints/{backbone}_stage1_best.weights.h5",
        monitor="val_accuracy",
        mode="max",
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )

    print(f"\n✅ Training Stage-1 for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=[csv_logger, ckpt_last, ckpt_best]
    )

    # Save final weights as official stage1
    out_weights = f"models/{backbone}_stage1.weights.h5"
    model.save_weights(out_weights)
    print(f"\n✅ Saved Stage-1 weights: {out_weights}")


if __name__ == "__main__":
    main()


