import cv2
import json
import os
import numpy as np

BASE_FOLDER = os.path.join("dataset", "objects_creation")
PREFERENCE_FILE_PATH = os.path.join("dataset", "class_preferences.json")
REFERENCE_FOLDER = os.path.join(BASE_FOLDER, "1_transparent_cropped")
AUGMENTED_FOLDER = os.path.join(BASE_FOLDER, "2_true_size")

AUGMENTATION = 70  # number of images per class
REFERENCE = (640, 85) # background pixels for 85 centimeters
scaling_factor = REFERENCE[0] / REFERENCE[1]  # pixels per centimeter

#--------------------------------------------------------------------------------

os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

with open(PREFERENCE_FILE_PATH, "r", encoding="utf-8") as f:
    preferences = json.load(f)

for class_folder in os.listdir(REFERENCE_FOLDER):

    print(f"Processing class: {class_folder}")

    reference_folder = os.path.join(REFERENCE_FOLDER, class_folder)
    current_folder = os.path.join(AUGMENTED_FOLDER, class_folder)
    os.makedirs(current_folder, exist_ok=True)

    files_list = os.listdir(reference_folder)

    for n in range(AUGMENTATION):

        file = files_list[n % len(files_list)]

        if file.endswith(".png"):

            print(f"Processing image {n}: {file}")

            current_image_path = os.path.join(reference_folder, file)
            current_image = cv2.imread(current_image_path, cv2.IMREAD_UNCHANGED)

            if current_image.shape[0] > current_image.shape[1]:
                main_dimension = 0
            else:
                main_dimension = 1
            
            index = np.random.randint(0, len(preferences[class_folder]["reference_size"]))
            
            new_main_size = int(preferences[class_folder]["reference_size"][index] * scaling_factor)
            current_scaling = new_main_size / current_image.shape[main_dimension]
            new_secondary_size = int(current_image.shape[not(main_dimension)] * current_scaling)

            if main_dimension:
                new_size = (new_main_size, new_secondary_size)
            else:
                new_size = (new_secondary_size, new_main_size)

            resized_image = cv2.resize(current_image, new_size)

            new_file_name = os.path.basename(current_folder) + "_" + str(n).zfill(4) + ".png"
            new_image_path = os.path.join(current_folder, new_file_name)

            cv2.imwrite(new_image_path, resized_image)