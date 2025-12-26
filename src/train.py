import os
from tensorflow.keras.optimizers import Adam
from src.utils import load_config
from src.data_loader import get_data_generators
from src.model import build_model

def train_model():
    config = load_config()

    train_generator, val_generator = get_data_generators(config)
    model = build_model(config)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=config["training"]["epochs"]
    )

    os.makedirs("outputs/models", exist_ok=True)
    model.save("outputs/models/mask_detection_model.h5")

    return history