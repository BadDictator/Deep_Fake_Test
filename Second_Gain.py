import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, LSTM, TimeDistributed,
                                     Bidirectional, GRU, Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation function
def image_augment(X_images, y_images):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    return datagen.flow(X_images, y_images, batch_size=32)

# CNN for Image-based Deep Fake Detection.
def create_cnn(input_shape=(128, 128, 3)):

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (Fake/Real)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# LSTM for Audio-based Deep Fake Detection.
def create_lstm(input_shape=(100, 20)):

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(GRU(64)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification (Fake/Real)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Combined Model for Both Image and Audio (Using both CNN and LSTM).
def create_combined_model(image_shape=(128, 128, 3), audio_shape=(100, 20)):

    # CNN for image processing
    cnn_input = Input(shape=image_shape, name="image_input")
    cnn_model = create_cnn(image_shape)
    cnn_output = cnn_model(cnn_input)

    # LSTM for audio processing
    lstm_input = Input(shape=audio_shape, name="audio_input")
    lstm_model = create_lstm(audio_shape)
    lstm_output = lstm_model(lstm_input)

    # Combine the outputs
    combined = tf.keras.layers.concatenate([cnn_output, lstm_output])
    combined_dense = Dense(64, activation='relu')(combined)
    combined_output = Dense(1, activation='sigmoid', name="combined_output")(combined_dense)

    # Create the full model
    model = Model(inputs=[cnn_input, lstm_input], outputs=combined_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example Training Workflow (with image augmentation)
if __name__ == "__main__":
    # Assuming you have preprocessed datasets
    # For images: `X_images`, `y_images`
    # For audio: `X_audio`, `y_audio`
    # For combined training: `[X_images, X_audio]`, `y_combined`

    # Mock dataset shapes (replace with real data loading)
    import numpy as np
    X_images = np.random.rand(1000, 128, 128, 3)
    y_images = np.random.randint(0, 2, 1000)

    X_audio = np.random.rand(1000, 100, 20)
    y_audio = np.random.randint(0, 2, 1000)

    # Create models
    cnn_model = create_cnn()
    lstm_model = create_lstm()
    combined_model = create_combined_model()

    # Train CNN (Image-based) with augmentation
    train_generator = image_augment(X_images, y_images)
    cnn_model.fit(train_generator, epochs=10, validation_split=0.2)

    # Train LSTM (Audio-based)
    lstm_model.fit(X_audio, y_audio, epochs=10, batch_size=32, validation_split=0.2)

    # Train Combined Model
    combined_model.fit(
        [X_images, X_audio], y_images,  # Use the same labels for simplicity in this example
        epochs=10, batch_size=32, validation_split=0.2
    )
