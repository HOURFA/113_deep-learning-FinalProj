import sys
import gc
sys.path.insert(0, "/home/rfa/NTUST/LAB/A-Yolom")
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f_v2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from memory_profiler import memory_usage
import time
import os

class ModelEvaluator:
    def __init__(self):
        self.source = '/home/rfa/NTUST/LAB/A-Yolom/ultralytics/datasets/small/images/val/b1cd1e94-26dd524f.jpg'
        self.data_sets = 'ultralytics/datasets/bdd-multi.yaml'
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.results = {
            'object_detection': [],
            'seg1_miou': [],
            'seg2_iou': [],
            'speed': [],
            'memory': []
        }
        # 定義要評估的模型列表
        self.models = [
            'ultralytics/v4.pt',
            'ultralytics/run/multi2/step_12_finetune2/weights/last.pt',
            'ultralytics/v4.engine',
            'ultralytics/run/multi2/step_12_finetune2/weights/last.engine'
        ]
        
    def cleanup_memory(self):
        """清理系統和GPU記憶體"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def predict_single_image(self, model_path):
        """執行單張圖片預測"""
        try:
            is_tensorrt = model_path.endswith((".engine", ".onnx"))
            model = YOLO(model_path)
            results = model.predict(
                source=self.source,
                device=self.device,
                batch=1,
                name='test',
                save=False,
                conf=0.6,
                iou=0.001,
                show_labels=True,
                boxes=True,
                show=True,
                TensorRt=is_tensorrt,
                task='multi'
            )
            return results
        finally:
            if 'model' in locals():
                del model
                self.cleanup_memory()

    def validate_model(self, model_path):
        """驗證模型並收集性能指標"""
        try:
            print(f"正在驗證模型: {model_path}")
            model = YOLO(model_path)
            
            # 執行驗證
            metrics, seg, speed = model.val(
                data=self.data_sets,
                device=self.device,
                task='multi',
                name='val',
                iou=0.6,
                conf=0.001,
                classes=[2, 3, 4, 9, 10, 11],
                combine_class=[2, 3, 4, 9],
                single_cls=True,
                speed=True
            )
            
            # 測量記憶體使用
            mem_usage = memory_usage(
                (self.predict_single_image, (model_path,), {}),
                interval=0.05,
                include_children=True,
                max_usage=False,
                retval=False
            )
            
            # 儲存結果
            self.results['object_detection'].append(metrics[0].box.map50)
            self.results['seg1_miou'].append(seg['seg-drivable']['mIoU'].avg)
            self.results['seg2_iou'].append(seg['seg-lane']['IoU'].avg)
            self.results['speed'].append(speed)
            self.results['memory'].append(sum(mem_usage) / len(mem_usage))
            
        except Exception as e:
            print(f"驗證過程發生錯誤: {str(e)}")
            raise
        finally:
            if 'model' in locals():
                del model
                self.cleanup_memory()

    def load_existing_results(self):
        """從CSV檔案載入現有結果"""
        if os.path.exists('results_ori_pruned_quantized.csv'):
            df = pd.read_csv('results_ori_pruned_quantized.csv')
            self.results['object_detection'] = df['Object Detection mAP50'].tolist()
            self.results['seg1_miou'] = df['Segmentation1 mIoU'].tolist()
            self.results['seg2_iou'] = df['Segmentation2 IoU'].tolist()
            self.results['speed'] = df['FPS'].tolist()
            self.results['memory'] = df['Memory Usage'].tolist()
            return len(self.results['object_detection'])
        return 0

    def save_results(self):
        """儲存結果到CSV"""
        df = pd.DataFrame({
            'Segmentation1 mIoU': self.results['seg1_miou'],
            'Segmentation2 IoU': self.results['seg2_iou'],
            'Object Detection mAP50': self.results['object_detection'],
            'FPS': self.results['speed'],
            'Memory Usage': self.results['memory']
        })
        df.to_csv('results_ori_pruned_quantized.csv', index=False)

    def plot_results(self):
        """繪製結果圖表"""
        x_labels = ['original', 'pruned', 'quantized', 'pruned&quantized']
        colors = ['royalblue', 'palegreen', 'lightcoral', 'cyan']
        
        # 性能指標圖
        plt.figure(figsize=(18, 5))
        metrics = [
            ('object_detection', 'Object Detection mAP50', 'mAP50'),
            ('seg1_miou', 'Drivable Area Segmentation Performance', 'mIoU'),
            ('seg2_iou', 'Lane Segmentation Performance', 'IoU')
        ]
        
        for idx, (key, title, ylabel) in enumerate(metrics, 1):
            plt.subplot(1, 3, idx)
            bars = plt.bar(x_labels[:len(self.results[key])], self.results[key], 
                          color=colors[:len(self.results[key])], width=0.25)
            plt.title(title)
            plt.xlabel('Model')
            plt.ylabel(ylabel)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x(), yval, f"{yval:.2f}", va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.close()
        
        # 速度和記憶體使用圖
        plt.figure(figsize=(12, 5))
        speed_memory = [
            ('speed', 'Inference Speed', 'FPS'),
            ('memory', 'Average Memory Usage', 'MiB')
        ]
        
        for idx, (key, title, ylabel) in enumerate(speed_memory, 1):
            plt.subplot(1, 2, idx)
            bars = plt.bar(x_labels[:len(self.results[key])], self.results[key], 
                          color=colors[:len(self.results[key])], width=0.25)
            plt.title(title)
            plt.xlabel('Model')
            plt.ylabel(ylabel)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x(), yval, f"{yval:.2f}", va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_metrics_speed_memory.png')
        plt.close()

    def run_evaluation(self):
        """執行評估流程"""
        # 載入現有結果
        completed_models = self.load_existing_results()
        print(f"已完成 {completed_models} 個模型的驗證")
        
        # 從未完成的模型開始驗證
        for i in range(completed_models, len(self.models)):
            try:
                print(f"開始驗證第 {i+1}/{len(self.models)} 個模型: {self.models[i]}")
                self.validate_model(self.models[i])
                self.save_results()  # 每次驗證後儲存結果
                print(f"完成驗證模型: {self.models[i]}")
            except Exception as e:
                print(f"驗證模型 {self.models[i]} 時發生錯誤: {str(e)}")
                break
            finally:
                self.cleanup_memory()
        
        # 繪製結果圖表
        self.plot_results()

def main():
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()

if __name__ == '__main__':
    main()
