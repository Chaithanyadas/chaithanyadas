import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

# Set parameters
img_width, img_height = 128, 128  # Define the size of the images
batch_size = 32
num_classes = 10  # Adjust based on your dataset
input_folder = '/content/drive/MyDrive/Colab Notebooks/New folder (1)/New folder/OutputFolder1/segmented image/temp_class'  # Change this to your folder
output_csv = '/content/drive/MyDrive/Colab Notebooks/New folder (1)/New folder/features.csv'  # Output CSV file name

# Define a simple CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer
    return model

# Create the CNN model
cnn_model = create_cnn_model((img_width, img_height, 3))
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# If you have a labeled dataset, you can use ImageDataGenerator to load it
# Here we assume you want to extract features without training; use this step for feature extraction directly.
# You can skip model training if your focus is solely on feature extraction from your dataset.

# Train your model here if you have labeled data
# For demonstration purposes, let's assume the model has been trained...

# Function to extract features from an image
def extract_features(image_path):
    # Load the image
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = cnn_model.predict(img_array)  # Extract features
    return features.flatten()  # Flatten the feature array

# List to hold the features
features_list = []
image_names = []

# Iterate through the images in the input folder
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        image_path = os.path.join(input_folder, image_name)
        features = extract_features(image_path)
        features_list.append(features)
        image_names.append(image_name)

# Create a DataFrame to save features
features_df = pd.DataFrame(features_list)
features_df['image_name'] = image_names

# Save the features to a CSV file
features_df.to_csv(output_csv, index=False)