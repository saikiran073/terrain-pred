import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

def create_data_generators():
    # Create train data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 20% validation split
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Create validation data generator
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    return train_datagen, valid_datagen

def create_model(num_classes):
    # Create base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(data_dir):
    # Create data generators
    train_datagen, valid_datagen = create_data_generators()
    
    # Create train generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Create validation generator
    validation_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print("Class indices:", train_generator.class_indices)
    
    # Create and compile model
    model = create_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Print TensorFlow version
    print("TensorFlow version:", tf.__version__)
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Specify your dataset directory
    data_dir = "/kaggle/input/terrain-dataset/terrain dataset"
    
    # Train the model
    model, history = train_model(data_dir)
    
    # Save the final model
    model.save('final_model.keras')
    