import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from ResNet50Model import res_net_50
from ResNet50Model2 import res_unet

NUM_CLASSES = 4
EPOCHS = 5
BATCH_SIZE = 16

# Definition of Dice's coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


# Simulate data
X = np.random.rand(1000, 224, 224, 3)  # Replace with preprocessed images
#simulate segmentation masks
Y = np.random.randint(0, NUM_CLASSES, 1000)  # Simulate labels
print(Y)
Y = to_categorical(Y, num_classes=NUM_CLASSES)  # One-hot encoding
print(Y)


# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create res_unet model
model = res_unet(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)
model.compile(optimizer=SGD(learning_rate=0.000062), loss='categorical_crossentropy',
              metrics=['accuracy'])  # Editar los hiperparámetros con los propuestos en el artículo

# Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)

# Compute Dice coefficient for each epoch on the validation set
dice_scores = []
for epoch in range(len(history.history['val_loss'])):
    val_preds = model.predict(X_val, batch_size=32)
    val_preds = np.argmax(val_preds, axis=1)  # Convertir probabilidades a etiquetas
    y_val_labels = np.argmax(Y_val, axis=1)  # Convertir one-hot a etiquetas
    dice = dice_coefficient(y_val_labels, val_preds)
    dice_scores.append(dice)

# Plot Dice coefficient evolution
plt.figure(figsize=(10, 5))
plt.plot(dice_scores, label='Dice Coefficient')
plt.title('Dice Coefficient Evolution')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.show()

# Plot losses to evaluate overfitting
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
