from ultralytics import YOLO

# Load a standard YOLO26 model
model = YOLO("yolo11n.pt")

# Export to TFLite format with INT8 quantization
# The 'int8' argument triggers Post-Training Quantization
# 'data' provides the calibration dataset needed for mapping values

model.export(format="tflite", int8=True, data="coco8.yaml")

#model.export(format="onnx", int8=True, data="coco8.yaml")
