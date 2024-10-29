import cv2
import os
import matplotlib.pyplot as plt

def preprocess_and_display_images(input_folder, output_folder):
    # Set up CLAHE and median filter parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    median_filter_size = 3  # Adjust filter size as needed for noise reduction

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through the main folder and subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)

            # Construct the output path
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            output_path = os.path.join(output_subfolder, file.replace('.dcm', '_preprocessed.png').replace('.jpg', '_preprocessed.png'))

            # Read the image in grayscale
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue  # Skip non-image files

            # Apply CLAHE for contrast enhancement
            clahe_image = clahe.apply(image)

            # Apply Median filter for noise removal
            median_filtered_image = cv2.medianBlur(clahe_image, median_filter_size)

            # Save the preprocessed image to the output path
            cv2.imwrite(output_path, median_filtered_image)

            # Display both the input and processed images
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Preprocessed Image (CLAHE + Median Filter)")
            plt.imshow(median_filtered_image, cmap='gray')
            plt.axis('off')

            plt.show()