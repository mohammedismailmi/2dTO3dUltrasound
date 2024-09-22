import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Define the directory where your images are stored
image_dir = r'C:\Users\DELLL\Downloads\FETAL_PLANES_ZENODO\Images'
image_height, image_width = 128, 128  # Assuming images will be resized to this

# Function to load and preprocess images from the directory
def load_images(image_dir, image_height, image_width):
    image_files = os.listdir(image_dir)
    images = []

    # Loop through all image files in the directory
    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        img = load_img(img_path, target_size=(image_height, image_width))  # Load and resize the image
        img_array = img_to_array(img)  # Convert to NumPy array
        images.append(img_array)

    return np.array(images)

# Load images from the directory
X_data = load_images(image_dir, image_height, image_width)

# Assuming you want to work with pairs of successive images, we'll create pairs
num_samples = len(X_data) - 1
X_train = np.zeros((num_samples, image_height, image_width, 2))  # 2 channels for successive frames

for i in range(num_samples):
    # Create a pair of successive images
    X_train[i, :, :, 0] = X_data[i].mean(axis=-1)  # Use mean for grayscale channel from RGB
    X_train[i, :, :, 1] = X_data[i + 1].mean(axis=-1)

# Sample labels for motion prediction (replace with actual labels)
y_train = np.random.rand(num_samples, 3)  # x, y, z motion labels

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, image_height * image_width * 2)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, image_height * image_width * 2)).reshape(X_test.shape)

# Check the shape of X_train and X_test
print(f"Before reshaping: X_train shape: {X_train.shape}")

# Reshape to make sure the input fits the model
X_train = X_train.reshape(-1, image_height, image_width, 2)
X_test = X_test.reshape(-1, image_height, image_width, 2)

print(f"After reshaping: X_train shape: {X_train.shape}")

# Define the CNN model for motion prediction
def build_model(input_shape):
    model = models.Sequential()

    # Convolutional layers for feature extraction
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add dropout layer for regularization
    model.add(layers.Dropout(0.2))

    # Flatten the layers and add dense layers for prediction
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3))  # Predict x, y, z translation or probe motion

    return model

# Build and compile the model
input_shape = (image_height, image_width, 2)
model = build_model(input_shape)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Predictions on test set
predictions = model.predict(X_test)
print(predictions[:5])

# Calculate and print the distances for the predictions
for i, (x, y, z) in enumerate(predictions):
    distance = np.sqrt(x**2 + y**2 + z**2)
    print(f"Prediction {i + 1}: The image has been shifted by approximately {distance:.4f} units in 3D space.")

