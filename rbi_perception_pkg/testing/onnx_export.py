from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo12n.pt")

# Export the model to ONNX format
model.export(format="onnx",
             imgsz=(960, 600),
             simplify=True,
             opset=16)  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolo12n.onnx")