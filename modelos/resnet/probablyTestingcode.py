import numpy as np
import matplotlib.pyplot as plt

# Dice coefficient calculation for test evaluation
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()  # Flatten the true labels
    y_pred_f = y_pred.flatten()  # Flatten the predicted labels
    intersection = np.sum(y_true_f * y_pred_f)  # Compute the intersection
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Simulated test dataset (replace this with actual test data)
X_test = np.random.rand(200, 224, 224, 3)  # Preprocessed test images
Y_test = np.random.randint(0, 2, 200)      # Binary labels (0 or 1)
Y_test = np.eye(2)[Y_test]                 # One-hot encoding for 2 classes

# Load the trained model (replace this with your model if saved)
# For example: from tensorflow.keras.models import load_model
# model = load_model('path_to_trained_model.h5')

# Make predictions
test_preds = model.predict(X_test, batch_size=32)
test_preds_labels = np.argmax(test_preds, axis=1)  # Convert probabilities to labels
y_test_labels = np.argmax(Y_test, axis=1)         # Convert one-hot to labels

# Calculate the Dice coefficient for the test set
test_dice = dice_coefficient(y_test_labels, test_preds_labels)
print(f"Dice Coefficient on Test Set: {test_dice:.4f}")

# Visualize test results
num_examples = 5  # Number of test images to display
for i in range(num_examples):
    plt.figure(figsize=(5, 5))
    plt.imshow(X_test[i])  # Display test image
    plt.title(f"True Label: {y_test_labels[i]}, Prediction: {test_preds_labels[i]}")
    plt.axis('off')
    plt.show()