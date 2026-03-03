import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input

# ==============================
# Load Dataset
# ==============================

train_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/train',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(128, 128)
)

test_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/test',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(128, 128)
)

# ==============================
# Normalize Images
# ==============================

def normalize(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(normalize)
test_ds = test_ds.map(normalize)

# ==============================
# Build CNN Model
# ==============================

model = Sequential([
    Input(shape=(128,128,3)),

    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# Train Model
# ==============================

history = model.fit(
    train_ds,
    epochs=7,
    validation_data=test_ds
)

# ==============================
# Save Model
# ==============================

model.save("my_model.keras")

print("Model training complete and saved successfully!")