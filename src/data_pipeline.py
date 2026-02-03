import tensorflow as tf
from config import TRAIN_DIR, VALID_DIR, IMG_SIZE, BATCH_SIZE

print("✅ Loading Plant Disease Dataset Pipeline...\n")

# -------------------------------
# Step 1: Load Training Dataset
# -------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",   # multi-class labels
    shuffle=True
)

# -------------------------------
# Step 2: Load Validation Dataset
# -------------------------------
valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# -------------------------------
# Step 3: Print Class Info
# -------------------------------
class_names = train_ds.class_names
print("🌿 Total Classes:", len(class_names))
print("📌 First 5 Classes:", class_names[:5])

# -------------------------------
# Step 4: Normalize Images
# -------------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

# -------------------------------
# Step 5: Performance Optimization
# -------------------------------
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Step 6: Check One Batch Shape
# -------------------------------
for images, labels in train_ds.take(1):
    print("\n✅ Batch Image Shape:", images.shape)
    print("✅ Batch Label Shape:", labels.shape)

print("\n🎉 Data Pipeline Ready for Training!")
