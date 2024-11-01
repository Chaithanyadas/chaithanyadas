import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 1: Define the Improved IAF-UNET architecture
def build_improved_iaf_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = layers.LeakyReLU(alpha=0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), padding='same')(c1)
    c1 = layers.LeakyReLU(alpha=0.1)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), padding='same')(p1)
    c2 = layers.LeakyReLU(alpha=0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), padding='same')(c2)
    c2 = layers.LeakyReLU(alpha=0.1)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), padding='same')(p2)
    c3 = layers.LeakyReLU(alpha=0.1)(c3)
    c3 = layers.Conv2D(256, (3, 3), padding='same')(c3)
    c3 = layers.LeakyReLU(alpha=0.1)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), padding='same')(p3)
    c4 = layers.LeakyReLU(alpha=0.1)(c4)
    c4 = layers.Conv2D(512, (3, 3), padding='same')(c4)
    c4 = layers.LeakyReLU(alpha=0.1)(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), padding='same')(p4)
    c5 = layers.LeakyReLU(alpha=0.1)(c5)
    c5 = layers.Conv2D(1024, (3, 3), padding='same')(c5)
    c5 = layers.LeakyReLU(alpha=0.1)(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), padding='same')(u6)
    c6 = layers.LeakyReLU(alpha=0.1)(c6)
    c6 = layers.Conv2D(512, (3, 3), padding='same')(c6)
    c6 = layers.LeakyReLU(alpha=0.1)(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), padding='same')(u7)
    c7 = layers.LeakyReLU(alpha=0.1)(c7)
    c7 = layers.Conv2D(256, (3, 3), padding='same')(c7)
    c7 = layers.LeakyReLU(alpha=0.1)(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), padding='same')(u8)
    c8 = layers.LeakyReLU(alpha=0.1)(c8)
    c8 = layers.Conv2D(128, (3, 3), padding='same')(c8)
    c8 = layers.LeakyReLU(alpha=0.1)(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), padding='same')(u9)
    c9 = layers.LeakyReLU(alpha=0.1)(c9)
    c9 = layers.Conv2D(64, (3, 3), padding='same')(c9)
    c9 = layers.LeakyReLU(alpha=0.1)(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)  # Binary segmentation

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Define input shape and build the model
input_shape = (256, 256, 1)  # Example input shape
model = build_improved_iaf_unet(input_shape)
model.summary()