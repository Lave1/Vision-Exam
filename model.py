import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN model
num_classes = len(os.listdir("C:/Users/Marco/Desktop/Scuola ITS/robe Git/Vision-Exam/immagini"))
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load and split the dataset
base_dir = 'C:/Users/Marco/Desktop/Scuola ITS/robe Git/Vision-Exam/immagini'
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1
)

# Evaluate the model on the test set
test_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get predictions on the test set
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=train_generator.class_indices.keys()))

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Example of predicting a single image
image_to_predict = "C:/Users/Marco/Desktop/Scuola ITS/robe Git/Vision-Exam/immagini_di_prova/bottiglietta di plastica/bottiglietta di plastica_204.jpg"
class_names = ["borraccia", "bottiglia di plastica", "bottiglia di vetro", "bottiglietta di plastica", "cuffie", "headset", "keyboard", "mouse", "smartphone"]

prediction_result = predict_single_image(model, image_to_predict, class_names)

for key, value in prediction_result.items():
    if key == "Confidence levels":
        print(f"{key}:")
        for class_name, confidence_level in value.items():
            print(f"  {class_name}: {confidence_level:.2f}%")
    else:
        print(f"{key}: {value}")
