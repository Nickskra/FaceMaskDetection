from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(config):
    img_path = config["dataset"]["img_path"]
    batch_size = config["training"]["batch_size"]
    img_size = tuple(config["model"]["image_size"])

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        img_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        img_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator