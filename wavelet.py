import os
import cv2
import numpy as np
import pywt
import pandas as pd

def load_image(image_path):
    """Load the image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def extract_wavelet_features(image):
    """Extract wavelet features from the given image."""
    # Apply Discrete Wavelet Transform (DWT)
    coeffs = pywt.dwt2(image, 'haar')  # You can change 'haar' to other wavelet types
    cA, (cH, cV, cD) = coeffs
    
    # Extract features
    features = {
        'cA_mean': np.mean(cA),
        'cA_std': np.std(cA),
        'cH_mean': np.mean(cH),
        'cH_std': np.std(cH),
        'cV_mean': np.mean(cV),
        'cV_std': np.std(cV),
        'cD_mean': np.mean(cD),
        'cD_std': np.std(cD),
    }
    
    return features

def extract_features_from_folder(input_folder, output_csv):
    """Extract wavelet features from all images in the input folder and save them to a CSV file."""
    # List to store the features for each image
    features_list = []

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
            image_path = os.path.join(input_folder, filename)
            image = load_image(image_path)
            
            if image is not None:  # Ensure image is loaded correctly
                features = extract_wavelet_features(image)
                features['filename'] = filename  # Add filename to features
                features_list.append(features)
            else:
                print(f"Could not load image: {filename}")

    # Create a DataFrame from the features list
    features_df = pd.DataFrame(features_list)

    # Save the DataFrame to a CSV file
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")