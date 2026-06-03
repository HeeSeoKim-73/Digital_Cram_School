from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=10,
    imgsz=416,
    batch=4,
    name="phone_detection_model",
    patience=5
)
