import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from ResNet50Model import res_net_50
from ResNet50Model2 import res_unet
import nibabel as nib

NUM_CLASSES = 4
EPOCHS = 2
BATCH_SIZE = 16

# Define input directories
input_images_dir = '../../DataBase/DataSet_Preprocessed_RGB_clear/'
input_masks_dir = '../../DataBase/MASK_RESIZED/'

# Definition of Dice's coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


# Function to load and preprocess images from NIfTI files (no conversion to RGB)
def load_image(image_path):
    img = nib.load(image_path)  # Load NIfTI file
    img_data = img.get_fdata()  # Get image data
    return img_data

# Function to load and preprocess segmentation masks from NIfTI files (without resizing)
def load_and_preprocess_mask(mask_path):
    mask = nib.load(mask_path)  # Load NIfTI file
    mask_data = mask.get_fdata()  # Get mask data
    # Convert mask values to one-hot encoding
    mask_one_hot = np.array([to_categorical(mask_data[:, :, i], num_classes=NUM_CLASSES) for i in range(mask_data.shape[2])])
    return mask_one_hot

# Load images and masks into arrays
image_files = [f for f in os.listdir(input_images_dir) if f.endswith('.nii.gz')]
mask_files = [f for f in os.listdir(input_masks_dir) if f.endswith('.nii.gz')]

# Ensure that image and mask files correspond
image_files.sort()
mask_files.sort()

# Define the range of images to load (e.g., from index 10 to index 30)
start_idx = 10
end_idx = 20

# Select only the files within the specified range
selected_image_files = image_files[start_idx:end_idx]
selected_mask_files = mask_files[start_idx:end_idx]

# Initialize lists to hold image and mask data
X = []
Y = []

# Load and preprocess all images and masks
for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(input_images_dir, img_file)
    mask_path = os.path.join(input_masks_dir, mask_file)
    
    # Load the image and mask
    image_data = load_image(img_path)  # No need to convert to RGB since they're already in that format
    mask_data = load_and_preprocess_mask(mask_path)
    
    # Append to the lists
    X.append(image_data)
    Y.append(mask_data)

# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Split data into training and validation sets (80% train, 20% validation)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Verifica las formas de los datos
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"Y_val shape: {Y_val.shape}")

# Create res_unet model
skip_layers = ["conv2d_62", "conv2d_83", "conv2d_92"]  
model = res_unet(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, skip_layer_names=skip_layers)
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
