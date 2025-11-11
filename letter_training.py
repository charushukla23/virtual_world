import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ======================
# Load EMNIST letters dataset
# ======================
print("Loading dataset...")
data = scipy.io.loadmat('E:/virtual_world/matlab/emnist-letters.mat')
dataset = data['dataset']

# Extract train/test images and labels
X_train = dataset['train'][0, 0]['images'][0, 0]
y_train = dataset['train'][0, 0]['labels'][0, 0].flatten()
X_test = dataset['test'][0, 0]['images'][0, 0]
y_test = dataset['test'][0, 0]['labels'][0, 0].flatten()

# Reshape to (28, 28)
X_train = X_train.reshape((-1, 28, 28))
X_test = X_test.reshape((-1, 28, 28))

# Fix orientation (transpose) and invert colors
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))
X_train = 255 - X_train
X_test = 255 - X_test

# Normalize images
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Add channel dimension
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Labels are 1-based, shift to 0-based (0–25 for 26 letters)
y_train -= 1
y_test -= 1

# Create tf.data datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(20000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128)

# ======================
# Build CNN model
# ======================
IMG_SIZE = 28
NUM_CLASSES = 26

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# Train model
# ======================
EPOCHS = 20  # more epochs for better accuracy
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
)

# ======================
# Save model and weights
# ======================
model.save('emnist_letters_model.keras')   # full model
model.save_weights('emnist_letters.weights.h5')  # only weights

# ======================
# Plot training curves & save
# ======================
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the figure
graph_filename = 'training_curves.png'
plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
print(f"Training curves saved as {graph_filename}")

# Show the figure
plt.show()
