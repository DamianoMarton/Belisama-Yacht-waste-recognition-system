import os
import cv2
import json
import shutil
import numpy as np

BASE_FOLDER = "objects_creation"
PREFERENCE_FILE_PATH = "class_preferences.json"
REFERENCE_FOLDER = os.path.join(BASE_FOLDER, "2_true_size")
AUGMENTED_FOLDER = os.path.join(BASE_FOLDER, "3_color_filtered")

FILTER_STRENGTH = 0.1 # 0 - 1
BRIGHTNESS_STD = 20 # standard deviation of brightness variation

color_dictionary = {
    "blue": [1, -1, -1],
    "red": [-1, -1, 1],
    "green": [-1, 1, -1],
    "yellow": [1, 1, -1],
    "white": [1, 1, 1],
    "black": [-1, -1, -1],
    "none": [0, 0, 0]
} # BGR format

#--------------------------------------------------------------------------------

with open(PREFERENCE_FILE_PATH, "r", encoding="utf-8") as f:
    preferences = json.load(f)

for class_folder in os.listdir(REFERENCE_FOLDER):
    print(f"Processing class: {class_folder}")

    reference_folder = os.path.join(REFERENCE_FOLDER, class_folder)
    current_folder = os.path.join(AUGMENTED_FOLDER, class_folder)

    colors = preferences[class_folder]["colors"]
    if "none" not in colors:
        colors.append("none")

    files_list =  os.listdir(reference_folder)

    for n, file in enumerate(files_list):
        print(f"Processing image {n}: {file}")

        current_image_path = os.path.join(reference_folder, file)
        current_image = cv2.imread(current_image_path, cv2.IMREAD_UNCHANGED)
        current_image = current_image.astype(np.float32) # to avoid overflow/underflow during operations

        current_color = colors[np.random.randint(0, len(colors))]
        color_filter = color_dictionary[current_color]

        if current_color != "none":
            brightness = np.random.normal(loc = 0, scale = BRIGHTNESS_STD)
            current_image[:, :, :3] = current_image[:, :, :3] + brightness
            clipped = np.clip(current_image[:, :, :3], 0, 255)
            current_image[:, :, :3] = clipped

            for i in range(3):
                filter_value = color_filter[i]
                if filter_value == 1:
                    current_image[:, :, i] = current_image[:, :, i] + (255 - current_image[:, :, i]) * FILTER_STRENGTH
                elif filter_value == -1:
                    current_image[:, :, i] = current_image[:, :, i] * (1 - FILTER_STRENGTH)
            
            current_image = np.clip(current_image, 0, 255).astype(np.uint8) # clip to valid range 0 -255
        new_file_name = os.path.basename(current_folder) + "_" + str(n).zfill(4) + ".png"
        new_image_path = os.path.join(current_folder, new_file_name)

        cv2.imwrite(new_image_path, current_image)