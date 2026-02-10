import tensorflow as tf
from src.config import IMG_SIZE


def get_preprocess_fn(backbone: str):
    b = backbone.lower()
    if b == "resnet50":
        return tf.keras.applications.resnet.preprocess_input
    if b == "densenet201":
        return tf.keras.applications.densenet.preprocess_input
    if b in ["efficientnetb0", "efficientnet_b0"]:
        return tf.keras.applications.efficientnet.preprocess_input
    if b in ["mobilenetv2", "mobilenet_v2"]:
        return tf.keras.applications.mobilenet_v2.preprocess_input
    raise ValueError(f"Unknown backbone: {backbone}")


def build_model(backbone: str, num_classes: int, dropout: float = 0.2):
    b = backbone.lower()

    if b == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    elif b == "densenet201":
        base = tf.keras.applications.DenseNet201(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    elif b in ["efficientnetb0", "efficientnet_b0"]:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    elif b in ["mobilenetv2", "mobilenet_v2"]:
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    base.trainable = False  # Stage-1 default

    preprocess = get_preprocess_fn(backbone)

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")

    # Our data pipeline already gives images in [0,1]
    x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(inputs)
    x = preprocess(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone}_stage1")
    return model, base
