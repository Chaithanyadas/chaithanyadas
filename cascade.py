import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import pandas as pd

def create_cascade_net(input_shape):
    """Create a simple Cascade-Net model for feature extraction."""
    model = models.Sequential()
    
    # First stage
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second stage
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third stage
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    return model

def load_and_preprocess_image(image_path):
    """Load and preprocess the image for model input."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (128, 128))  # Resize to match model input shape
    image = image / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

def extract_features(image_folder, output_csv):
    """Extract features from all images in the input folder and save them to a CSV file."""
    model = create_cascade_net(input_shape=(128, 128, 3))  # Define model
    features_list = []

    # Iterate through all image files in the input folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
            image_path = os.path.join(image_folder, filename)
            image = load_and_preprocess_image(image_path)
            
            # Extract features
            features = model.predict(image)
            features = features.flatten()  # Flatten to 1D array
            features_list.append({'filename': filename, 'features': features})

    # Convert features list to DataFrame and save to CSV
    features_df = pd.DataFrame(features_list)
    features_df['features'] = features_df['features'].apply(lambda x: x.tolist())  # Convert ndarray to list
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")
