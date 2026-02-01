from ultralytics import YOLO
import numpy as np

model = YOLO(r"trained_models/yolo/experiment_1/weights/best.pt")

n = np.random.randint(0, 999)
print(f"Predicting image number: {n}")
image_path = f"dataset/dataset/images/val/validation_image_{str(66).zfill(4)}.jpg"
results = model.predict(source=image_path, conf=0.25, device='cpu', save=True)

res = results[0]

for box in res.boxes:
    coords = box.xyxy[0].tolist() # bounding box coordinates
    conf = box.conf[0].item() # confidence
    cls_id = int(box.cls[0].item()) # clss id
    cls_name = model.names[cls_id] # class name

    print(f"Object: {cls_name} | Confidence: {conf:.2f} | Box: {coords}")

res.show()