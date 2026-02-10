from pathlib import Path
import tensorflow as tf

from src.data_pipeline import get_datasets
from src.config import IMG_SIZE, EPOCHS_STAGE1


def build_resnet50_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base.trainable = False  # Freeze the base (Stage 1)

    # 2) Build new model on top (classification head)
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Our pipeline outputs images in [0,1] after Rescaling(1./255)
    # ResNet preprocess expects [0,255], so convert back then preprocess
    x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(inputs)
    x = tf.keras.applications.resnet.preprocess_input(x)

    # Pass through frozen base
    x = base(x, training=False)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="ResNet50_TransferLearning")
    return model


if __name__ == "__main__":
    # --- Create folders (so saving never fails) ---
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("models/logs").mkdir(parents=True, exist_ok=True)

    print("\n✅ Step 1: Loading datasets...")
    train_ds, valid_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print(f"✅ Total classes: {num_classes}")

    print("\n✅ Step 2: Building ResNet50 (ImageNet pretrained) model...")
    model = build_resnet50_model(num_classes)

    print("\n✅ Step 3: Compiling model...")

    # Faster on Apple Silicon vs normal Adam warning
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
        ]
    )

    # --- Callbacks: auto-save + log ---
    # 1) Save model after EVERY epoch (latest)
    checkpoint_each_epoch = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/checkpoints/resnet50_last.weights.h5",
        save_best_only=False,
        save_weights_only=True,
        verbose=1
    )

    # 2) Save BEST model by validation accuracy
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/checkpoints/resnet50_best.weights.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # 3) Save metrics each epoch to CSV (for plots later)
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename="models/logs/resnet50_stage1_log.csv",
        append=False
    )

    callbacks = [checkpoint_each_epoch, checkpoint_best, csv_logger]

    print("\n✅ Step 4: Starting training (Stage 1: only new head trains)...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks
    )

    # Save final stage-1 model (the final epoch weights)
    model.save_weights("models/resnet50_stage1.weights.h5")
    print("\n✅ Weights saved to: models/resnet50_stage1.weights.h5")

