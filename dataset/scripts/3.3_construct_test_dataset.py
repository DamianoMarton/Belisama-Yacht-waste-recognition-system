import json
import os
from pydoc import text
import shutil
import cv2
from matplotlib import text
import numpy as np
from imfuncs import imfuncs # personal functions for image manipulation

"""
Main differences with training dataset construction:
- Rotate only if class preference allows it
- Objects positioned mainly in the lower extremum of the image 
    (every class has its probability to not being in the lower extremum)
- Use of algae masses if bottom-positioned algae are required
"""

BASE_FOLDER = "objects_creation"
PREFERENCE_FILE_PATH = "class_preferences.json"
MAIN_REFERENCE_FILE_PATH = "main_reference.json"
REFERENCE_FOLDER = os.path.join(BASE_FOLDER, "3_color_filtered")
BACKGROUND_FOLDER = "refined_backgrounds"
REFERENCE_MASSES_FOLDER = os.path.join(BASE_FOLDER, "4_masses")
TEST_IMAGES_FOLDER = os.path.join("dataset", "images", "test")
TEST_LABELS_FOLDER = os.path.join("dataset", "labels", "test")

with open(MAIN_REFERENCE_FILE_PATH, "r", encoding="utf-8") as f:
    main_reference = json.load(f)

TEST_DATASET_SIZE = 800  # total number of images in the test dataset

HORIZONTAL_EXTREMA = (main_reference["background"]["upper_left_limit"][0], 
                      main_reference["background"]["lower_right_limit"][0]) # horizontal pixel range
VERTICAL_EXTREMA = (main_reference["background"]["upper_left_limit"][1], 
                    main_reference["background"]["lower_right_limit"][1]) # vertical pixel range
MAIN_DIMENSIONS = main_reference["background"]["dimensions"]
DISTRIBUTIONS = {"number_of_elements": {"min": 0, "max": 3}, #uniform distribution
                 "additional_elements": {"p": 0.5}, #geometric distribution
                 "additional_algae": {"min": 5, "max": 30}, # uniform distribution
                 "p_masses": 0.2,  # probability of using masses in the image
                 "squeeze_factor": {"mean": 0.6, "std": 0.2}  # vertical squeeze factor for masses
                 } 

#--------------------------------------------------------------------------------

if os.path.exists(TEST_IMAGES_FOLDER):
    shutil.rmtree(TEST_IMAGES_FOLDER)
os.makedirs(TEST_IMAGES_FOLDER)

if os.path.exists(TEST_LABELS_FOLDER):
    shutil.rmtree(TEST_LABELS_FOLDER)
os.makedirs(TEST_LABELS_FOLDER)
#--------------------------------------------------------------------------------

with open(PREFERENCE_FILE_PATH, "r", encoding="utf-8") as f:
    preferences = json.load(f)

statistics = {"number_of_elements_per_image": {},
              "classes_distribution": {}}

for N in range(TEST_DATASET_SIZE):
    print(f"Creating test image {N+1}/{TEST_DATASET_SIZE}")
    
    images = []
    labels = []

    shadow_offset = (np.random.randint(-7,8), np.random.randint(-7,8))
    shadow_intensity = np.random.randint(50, 100)

    n_elements = np.random.uniform(DISTRIBUTIONS["number_of_elements"]["min"],
                                       DISTRIBUTIONS["number_of_elements"]["max"] + 1)
    n_elements = int(n_elements)
    additional_elements = np.random.geometric(DISTRIBUTIONS["additional_elements"]["p"]) - 1 # so that 0 is possible
    n_elements += additional_elements

    statistics["number_of_elements_per_image"][n_elements] = statistics["number_of_elements_per_image"].get(n_elements, 0) + 1

    for n in range(n_elements):

        canvas = np.zeros((MAIN_DIMENSIONS[1], MAIN_DIMENSIONS[0], 4), dtype=np.uint8)

        class_folder = np.random.choice(os.listdir(REFERENCE_FOLDER))
        reference_class_folder = os.path.join(REFERENCE_FOLDER, class_folder)
        statistics["classes_distribution"][class_folder] = statistics["classes_distribution"].get(class_folder, 0) + 1
        preferences_for_class = preferences[class_folder]

        bottom = np.random.binomial(1, preferences_for_class["data_augmentation"]["p_bottom"])

        if class_folder != "algae":

            files_list = os.listdir(reference_class_folder)
            chosen_file = np.random.choice(files_list)

            current_image_path = os.path.join(reference_class_folder, chosen_file)
            current_image = cv2.imread(current_image_path, cv2.IMREAD_UNCHANGED)

            resize_factor = np.random.uniform(preferences_for_class["data_augmentation"]["resize_min"],
                                            preferences_for_class["data_augmentation"]["resize_max"])
            resized_image = imfuncs.resize_image(current_image, resize_factor)

            if bottom:
                if preferences_for_class["data_augmentation"]["rotate"] == 90:
                    rotated_image = imfuncs.random_rotation(resized_image, only_90 = True)
                elif preferences_for_class["data_augmentation"]["rotate"] == 180:
                    rotated_image = imfuncs.random_rotation(resized_image, only_180 = True)
                else:
                    rotated_image = imfuncs.random_rotation(resized_image)
            else:
                rotated_image = imfuncs.random_rotation(resized_image)

            x_b, y_b, w_b, h_b = imfuncs.get_bounding_box(rotated_image)
            
            shadow = imfuncs.get_shadow(rotated_image, shadow_intensity)

            x_pos, y_pos = imfuncs.get_main_positions(HORIZONTAL_EXTREMA, VERTICAL_EXTREMA, rotated_image)
            if bottom:
                y_pos = VERTICAL_EXTREMA[1] - rotated_image.shape[0] - np.random.randint(0, 5)

            labels.append((preferences_for_class["label"], x_pos + x_b, y_pos + y_b, w_b, h_b))

            imfuncs.paste_rgba(canvas, shadow, x_pos + shadow_offset[0], y_pos + shadow_offset[1]) # offset shadow
            imfuncs.paste_rgba(canvas, rotated_image, x_pos, y_pos)

            images.append(canvas)

        else: # algae special case

            files_list = os.listdir(reference_class_folder)
            
            n_algae = np.random.randint(DISTRIBUTIONS["additional_algae"]["min"],
                                        DISTRIBUTIONS["additional_algae"]["max"] + 1)
            
            bottom = np.random.binomial(1, preferences_for_class["data_augmentation"]["p_bottom"])

            if bottom:
                squeeze = np.random.normal(loc = DISTRIBUTIONS["squeeze_factor"]["mean"], 
                                        scale = DISTRIBUTIONS["squeeze_factor"]["std"])
                squeeze = np.clip(squeeze, 0.1, 1.0)

            for _ in range(n_algae):

                chosen_file = np.random.choice(files_list)

                current_image_path = os.path.join(reference_class_folder, chosen_file)
                current_image = cv2.imread(current_image_path, cv2.IMREAD_UNCHANGED)

                resize_factor = np.random.uniform(preferences_for_class["data_augmentation"]["resize_min"],
                                                preferences_for_class["data_augmentation"]["resize_max"])
                resized_image = imfuncs.resize_image(current_image, resize_factor)

                rotated_image = imfuncs.random_rotation(resized_image)

                if bottom:
                    new_height = int(rotated_image.shape[0] * squeeze)
                    rotated_image = cv2.resize(rotated_image, (rotated_image.shape[1], new_height), 
                                            interpolation=cv2.INTER_AREA)

                shadow = imfuncs.get_shadow(rotated_image, shadow_intensity)

                if not imfuncs.get_bounding_box(canvas):
                    x_pos = np.random.randint(HORIZONTAL_EXTREMA[0],
                                              HORIZONTAL_EXTREMA[1] - rotated_image.shape[1])
                    y_pos = np.random.randint(VERTICAL_EXTREMA[0],
                                              VERTICAL_EXTREMA[1] - rotated_image.shape[0])
                else:
                    x_pos, y_pos = imfuncs.get_secondary_positions(HORIZONTAL_EXTREMA, VERTICAL_EXTREMA,
                                                                canvas, rotated_image, 
                                                                shadow_offset)
                
                if bottom:
                    y_pos = VERTICAL_EXTREMA[1] - rotated_image.shape[0] - np.random.randint(0, 5)
                
                """
                cv2.imshow("debug algae", canvas)
                cv2.waitKey(0)
                """

                imfuncs.paste_rgba(canvas, shadow, x_pos + shadow_offset[0], y_pos + shadow_offset[1])
                imfuncs.paste_rgba(canvas, rotated_image, x_pos, y_pos)

            

            x_b, y_b, w_b, h_b = imfuncs.get_bounding_box(canvas)

            labels.append((preferences_for_class["label"], x_b, y_b, w_b, h_b))
            images.append(canvas)

    background_file = np.random.choice(os.listdir(BACKGROUND_FOLDER))
    background_path = os.path.join(BACKGROUND_FOLDER, background_file)
    background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)

    global_offset = (np.random.randint(0, 4), np.random.randint(0, 4)) # global offset to simulate camera misalignment
    background = background[global_offset[1]:background.shape[0] - (4 -global_offset[1]),
                                global_offset[0]:background.shape[1]- (4 - global_offset[0])] # output 640x640
    
    if images:
        final_objects = imfuncs.manage_overlap(images)
        imfuncs.paste_rgba(background, final_objects, 0, 0)

    global_brightness = np.random.randint(-25, 25)
    global_contrast = np.random.uniform(0.8, 1.2)

    adjusted_image = cv2.convertScaleAbs(background, alpha=global_contrast, beta=global_brightness)
    blurred_image = cv2.blur(adjusted_image, (3,3))

    final_bgr = cv2.cvtColor(blurred_image, cv2.COLOR_BGRA2BGR)
    
    new_name = f"test_image_{N:04d}"
    cv2.imwrite(os.path.join(TEST_IMAGES_FOLDER, f"{new_name}.jpg"), final_bgr)

    with open(os.path.join(TEST_LABELS_FOLDER, f"{new_name}.txt"), "w") as f:
        for label in labels:
            class_id, x_b, y_b, w_b, h_b = label
            x_center = np.clip((x_b + w_b / 2.0) / MAIN_DIMENSIONS[0], 0.0, 1.0)
            y_center = np.clip((y_b + h_b / 2.0) / MAIN_DIMENSIONS[1], 0.0, 1.0)
            w_norm = np.clip(w_b / MAIN_DIMENSIONS[0], 0.0, 1.0)
            h_norm = np.clip(h_b / MAIN_DIMENSIONS[1], 0.0, 1.0)
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    # Debug: show bounding boxes
    if N % ((TEST_DATASET_SIZE - 1)//4) == 0:  # show 5 samples
        for label in labels:
            class_id, x_b, y_b, w_b, h_b = label
            cv2.rectangle(blurred_image, (int(x_b), int(y_b)), (int(x_b + w_b), int(y_b + h_b)), (255,0,0), 1)
            cv2.putText(blurred_image, str(class_id), (x_b, y_b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.imshow("debug", blurred_image)
        cv2.waitKey(0)

print("\nNumber of elements statistics in validation dataset:")
for n_elements, count in sorted(statistics["number_of_elements_per_image"].items()):
    print(f"{n_elements}: {count}")
print("\nClasses distribution in validation dataset:")
for class_name, count in sorted(statistics["classes_distribution"].items()):
    print(f"{class_name}: {count}")