from ultralytics import YOLO

# Load the model trained on our custom data
model = YOLO("best.pt")

model.export(format="tflite", int8=True, data="../custom.yaml")