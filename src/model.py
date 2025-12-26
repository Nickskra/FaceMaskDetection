from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

def build_model(config):
    img_size = tuple(config["model"]["image_size"])
    num_classes = config["model"]["num_classes"]

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=(*img_size, 3))
    )

    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(128, activation='relu')(head_model)
    head_model = Dense(128, activation='relu')(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(num_classes, activation='softmax')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    return model