import sys
sys.path.insert(0, r"C:\Users\abc78\desktop\Jetson\home\rfa\LAB\A-Yolom\ultralytics")
from ultralytics import YOLO
windows = False
window_path = r"C:\Users\abc78\desktop\Jetson" if windows else ""
model_path = window_path + "/home/rfa/LAB/A-Yolom/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml"
pre_train_model_path = window_path + "/home/rfa/LAB/A-Yolom/ultralytics/last.pt"
dataset_path = window_path + "/home/rfa/LAB/A-Yolom/ultralytics/datasets/bdd-multi.yaml"
device = 'CPU' if windows else 0
# # Load a model
model = YOLO(model_path, task='multi').load(pre_train_model_path)  # build a new model from YAML

#
#
# # Train the model
# model.train(data=r'C:\Users\abc78\YOLOv8-multi-task-main\ultralytics\datasets\test.yaml',device='CPU',batch=4,epochs=100,imgsz=(640,640),name='test',val=True,task='multi',classes=[0,1,2,3,4,5,6,7,8])
model.train(data=dataset_path,
            device=device,
            batch=4,
            epochs=100,
            imgsz=(640,640),
            name='test',
            val=True,
            task='multi',
            classes=[0,1,2,3,4,5,6,7,8,9,10,11],
            combine_class=[2,3,4,9],
            single_cls=False,
            plots = True)


