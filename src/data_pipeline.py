import tensorflow as tf
from src.config import TRAIN_DIR, VALID_DIR, IMG_SIZE, BATCH_SIZE


def get_datasets():
    print("✅ Loading datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"🌿 Classes found: {len(class_names)}")

    # Normalize 0–255 -> 0–1
    norm = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, class_names


if __name__ == "__main__":
    # This block runs only when you run data_pipeline.py directly
    train_ds, valid_ds, class_names = get_datasets()

    for images, labels in train_ds.take(1):
        print("✅ Batch Image Shape:", images.shape)
        print("✅ Batch Label Shape:", labels.shape)
