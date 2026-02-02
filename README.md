# Belisama Yacht Waste Recognition

**Synthetic Data Pipeline and YOLO-like Object Detector for Waste Recognition in Venice Workboats.**

This project develops an artificial vision system for automatic recognition of floating waste in the channels of Venice. The system is designed to operate on the conveyor belts of **Belisama Yacht** workboats, providing periodic reports on the type and quantity of waste collected.

---

## Table of Contents
- [Project Description](#project-description)
- [Repository Structure](#repository-structure)
- [Synthetic Dataset Pipeline](#synthetic-dataset-pipeline)
- [Object Detection Models](#object-detection-models)
- [Results](#results)
- [Contacts](#contacts)

---

## Project Description
Since no test videos are available to build the images to be analyzed, as the first boat of this class is expected to be ready by April 2026, we developed a pipeline for generating **realistic synthetic images**. The system analyzes frames extracted from the conveyor belt via an overhead camera.

The project compares two approaches:
1. **YOLOv11 Nano Fine-tuning**: for professional performance.
2. **Custom Minimal YOLO**: a model with 1.6M parameters developed from scratch to analyze YOLO mechanisms.

---

## Repository Structure

```text
├── dataset/
│   ├── dataset/                # Folder with the yaml file, where the synthetic dataset is going to be
│   ├── objects_creation/       # Folder for processing object images
│   ├── refined_backgrounds/    # Conveyor belt images used as backgrounds
│   ├── scripts/                # Python scripts to generate dataset (numerated)
│   ├── class_preferences.json  # Parameters (colors, size, rotation) for each class
│   └── main_reference.json     # Working area limits and dimensions
│
├── scripts/
│   ├── my_yolo/                # Custom model code (Loss, Metrics, Colab Notebook)
│   └── yolo/                   # Notebooks for YOLOv11 fine-tuning and test scripts
│
├── trained_models/
│   ├── my_yolo/                # Custom model weights (.pt)
│   └── yolo/                   # Fine-tuned YOLOv11 weights (.pt)
│
└── README.md
```

## Synthetic Dataset Pipeline
Data generation is managed by scripts in the dataset/scripts/ folder.

**Operating Logic**:

- **Configuration**: Realism parameters are defined in class_preferences.json (e.g., belt adhesion probability, rotation, scaling).
- **Preprocessing**: Object images are cropped, oriented horizontally, and filtered for color and brightness variations.
- **Augmentation**: Uniform blur is applied to reduce sharp overlay edges and simulate camera oscillations through dynamic background crops defined in main_reference.json.

After running the scripts in the sequence shown and fine-tuning the parameters for each class and script, your dataset will be ready for use.

## Object Detection Models
**YOLOv11 Nano**

We used transfer learning to adapt YOLOv11n to our specific scenario (13 waste classes + 1 "other" class).

- **Training Notebook**: Available in scripts/yolo/
- **Performance**: Precision 0.97, Recall 0.93.

**Custom Minimal YOLO-like**

Developed for internal research purposes, it uses:

- **Architecture**: Convolutional backbone (stride 2) and Neck with residual blocks (skip connections).
- **Grid**: 20x20 cells (up to 400 predictions, just one for each cell).
- **Loss**: Binary Cross Entropy (objectness), IoU Loss (bounding boxes) and Cross Entropy (classes).
- **Source Code**: Available in scripts/my_yolo/

## Results
The custom model achieved an F1-score of 0.69, demonstrating excellent localization capabilities, although it is affected by heavy occlusions in cases of overlapping objects. YOLOv11n, on the other hand, provides the robustness necessary for industrial deployment.

## Contacts
**Damiano Marton** - Belisama Yacht

📧 damiano.marton@studenti.unipd.it

📧 damianomarton@belisamayacht.it

🌐 www.belisamayacht.it
