import os
import cv2
import numpy as np
import pandas as pd

# Function to convert images to flattened arrays
def image_to_array(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (320, 240))  # Adjust size as needed
    return img.flatten()

# Define the path to your dataset
data_dir = 'Pictures'

# Create CSV files for each character
characters = os.listdir(data_dir)

for character in characters:
    character_dir = os.path.join(data_dir, character)
    csv_filename = f'{character}_data.csv'
    
    data = []
    for filename in os.listdir(character_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(character_dir, filename)
            img_array = image_to_array(image_path)
            data.append(img_array)

    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)
