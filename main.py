import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from PIL import Image

# Disabling oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

data_dir = pathlib.Path('./Dataset/Train')

# Counting images
image_count = len(list(data_dir.glob('*/*.png')))
print(f"Total images: {image_count}")

# List files (if needed)
# list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.png'))

# Training split
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32,
    class_names=['Cross', 'Zeroes'],
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32,
    class_names=['Cross', 'Zeroes'],
    label_mode="categorical"
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

num_classes = len(class_names)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        # Convert label tensor to numpy and get the scalar value
        label_index = labels[i].numpy().argmax()  # This assumes categorical labels

        plt.title(class_names[label_index])
        plt.axis("off")
# Building the model
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(180, 180, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Training the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

#loss
loss = history.history['loss']
val_loss = history.history['val_loss']

#epochs
epochs_range = range(epochs)

# Plotting graphs
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

path = r"./Dataset/Test"
predicted_data = {'ID': [], 'POS_1': [], 'POS_2': [], 'POS_3': [], 'POS_4': [], 'POS_5': [], 'POS_6': [], 'POS_7': [],
                  'POS_8': [], 'POS_9': []}
for filename in listdir(path):
    img = Image.open(path + f'/{filename}')
    predicted_data['ID'].append(int(filename.removesuffix('.png')))
    y1 = 59
    y2 = 176
    for i in range(3):
        x2 = 262
        x1 = 144
        for j in range(3):
            pos = 3 * i + j + 1
            img_temp = img.crop((x1, y1, x2, y2))
            img_temp = img_temp.resize((180, 180))
            img_temp = img_temp.convert('RGB')
            img2 = img_temp.convert('L')
            img2 = img2.crop((8, 10, 170, 170))
            img_array = np.array(img_temp)
            img2_array = np.array(img2)
            img_array = np.expand_dims(img_array, axis=0)
            blank = np.full_like(img2_array, 30)
            if np.array_equal(img2_array, blank):
                predicted_data[f'POS_{pos}'].append(2)
            else:
                pred = model.predict(img_array)
                prediction = np.argmax(pred)
                predicted_data[f'POS_{pos}'].append(prediction)
            x1 = x1 + 120
            x2 = x2 + 120
        y1 = y1 + 120
        y2 = y2 + 120
df = pd.DataFrame(predicted_data)
df['ID'] = pd.to_numeric(df['ID'])
df = df.sort_values("ID")
print(df)
df.to_csv('submission.csv', index=False)

