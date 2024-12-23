import sys
sys.path.insert(0, "/home/rfa/NTUST/LAB/A-Yolom")
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f_v2
# Load a model
model_path = "./ultralytics/run/multi2/step_12_finetune2/weights/last.pt"
#model_path = "./ultralytics/v4.pt"
model = YOLO(model_path)

# Export the model
# model.export(format='engine', device = 0, simplify = True, half=True,workspace=4,opset = 17)
# model.export(format='engine', device = 0, simplify = True, int8 = True, opset = 12, workspace = 8, batch = 1,half = True)

model.export(
    format="engine",
    device = 0,
    dynamic=True,  
    batch=1,  
    workspace=8,  
    int8=True,
    opset=12,
    simplify = True,
    data='ultralytics/datasets/bdd-multi_demo.yaml'  
)

# # Load the exported TensorRT INT8 model
# model = YOLO("yolov8n.engine", task="detect")

# # Run inference
# result = model.predict("https://ultralytics.com/images/bus.jpg")
