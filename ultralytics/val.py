import sys
sys.path.insert(0, "/home/rfa/NTUST/LAB/A-Yolom")
# 现在就可以导入Yolo类了
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f_v2
# model = YOLO('yolov8s-seg.pt')
# number = 3 #input how many tasks in your work
model_path = "./ultralytics/run/multi/step_13_finetune/weights/last.pt"
model = YOLO(model_path)  # 加载自己训练的模型# Validate the model
# metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[4],task='multi',name='v3-model-val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)  # no arguments needed, dataset and settings remembered

metrics, seg, speed= model.val(data='ultralytics/datasets/bdd-multi_demo.yaml',
                    device=0,task='multi',name='val',iou=0.6,conf=0.001, 
                    imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],
                    speed = True,single_cls=True)  # no arguments needed, dataset and settings remembered

print(speed)
# for i in range(number):
#     print(f'This is for {i} work')
#     print(metrics[i].box.map)    # map50-95
#     print(metrics[i].box.map50)  # map50
#     print(metrics[i].box.map75)  # map75
#     print(metrics[i].box.maps)   # a list contains map50-95 of each category