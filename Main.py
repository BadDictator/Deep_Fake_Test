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

# Features extracted in CNN: edges, textures, spatial relationships, representations of facial landmarks or objects (high-level patterns).
# Featues extracted in LSTM: temporal and acoustic features (pitch variations, rhythm, speaker characteristics, anomalies, or artifacts).
# CNN model Output: represented as a vector of numerical values.
# LSTM model Output: represented as a vector of numerical values.

# Combined input: Vector of numerical values from CNN and LSTM.
# Combined output: 0 or 1.

# CNN for Image-based Deep Fake Detection.
def create_cnn(input_shape=(128, 128, 3)):

    # Initialise a sequential model.
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), # Convolutional layer with 32 filters of size 3x3.
        MaxPooling2D((2, 2)), # Max pooling layer with window size 2x2.
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(), # Multi to 1 dimensional vector.
        Dense(128, activation='relu'), # 128 neurons fully connected, Rectified Linear Unit.
        Dropout(0.5), # Dropout to prevent overfitting.
        Dense(1, activation='sigmoid')  # Binary classification (Fake/Real)
        # 1 = Fake, 0 = Real.
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# LSTM for Audio-based Deep Fake Detection.
def create_lstm(input_shape=(100, 20)):

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape), # LSTM layer with 128 units. #
        # True ensures the return of the full sequence of outputs.
        Dropout(0.3), 
        Bidirectional(GRU(64)), # Gated recurrent Unit, both forward and backward.
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
    cnn_model = create_cnn(image_shape) # Create instance of CNN model.
    cnn_output = cnn_model(cnn_input)

    # LSTM for audio processing
    lstm_input = Input(shape=audio_shape, name="audio_input")
    lstm_model = create_lstm(audio_shape)
    lstm_output = lstm_model(lstm_input)

    # Combine the outputs
    combined = tf.keras.layers.concatenate([cnn_output, lstm_output])
    combined_dense = Dense(64, activation='relu')(combined) # 64 neurons and ReLU for better combination.
    combined_output = Dense(1, activation='sigmoid', name="combined_output")(combined_dense)

    # Create the full model
    model = Model(inputs=[cnn_input, lstm_input], outputs=combined_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Why CNN?: Local Feature Detection, Spatial Invariance, Feature Hierarchy, Computational Efficiency.
# Why not RNN?: Vanishing/exploding gradient problem. 
# Why not FCNs?: FCNs require processing all pixels in the image.

# Why LSTM? : Handling Sequential Data, Capturing Long-Term Dependencies, Robustness to Noise.
# Why not RNN?: Vanishing/exploding gradient problem.
# Why not FNN?: Feedforward Neural Networks cannot handle sequential data.
