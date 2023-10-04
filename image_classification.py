import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Data Preparation with Data Augmentation
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(train_images)

# Step 2: Model Creation with a Pre-trained Architecture (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Step 3: Model Training
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32, epochs=20,
                    validation_data=(test_images, test_labels))

# Step 4: Model Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

predicted_labels = np.argmax(model.predict(test_images), axis=1)
true_labels = np.argmax(test_labels, axis=1)
confusion_mtx = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:\n", confusion_mtx)
print("\nClassification Report:\n", classification_report(true_labels, predicted_labels))

# Step 5: Prediction
sample_images = test_images[:5]
sample_labels = true_labels[:5]

predicted_classes = model.predict(sample_images)
predicted_classes = np.argmax(predicted_classes, axis=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display the sample images and their predicted classes
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(sample_images[i])
    plt.title(f"True: {class_names[sample_labels[i]]}\nPredicted: {class_names[predicted_classes[i]]}")
    plt.axis('off')

plt.show()
