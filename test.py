import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore

# =========================
# 1. Data Preprocessing
# =========================

train_dir = "data/train"
val_dir = "data/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# =========================
# 2. Model Architecture (Deep CNN)
# =========================

model = models.Sequential()

# Block 1
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3))) #type: ignore
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Block 2
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Block 3
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Block 4 (Deeper)
model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Fully Connected Layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(train_data.num_classes, activation='softmax'))

# =========================
# 3. Compile Model
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 4. Callbacks (Important for DL)
# =========================

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3),
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
]

# =========================
# 5. Training
# =========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks
)

# =========================
# 6. Evaluation
# =========================

loss, acc = model.evaluate(val_data)
print(f"Validation Accuracy: {acc:.4f}")

# =========================
# 7. Prediction Example
# =========================

import numpy as np
from tensorflow.keras.preprocessing import image #type: ignore

img_path = "test.jpg"
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print("Predicted class:", np.argmax(prediction))