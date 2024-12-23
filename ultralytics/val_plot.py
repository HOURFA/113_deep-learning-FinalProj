import sys
sys.path.insert(0, "/home/rfa/NTUST/LAB/A-Yolom")
import gc
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f_v2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def cleanup_memory():
    """清理 GPU 和系統記憶體"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_existing_results():
    """讀取已存在的 results.csv 檔案"""
    if os.path.exists('results.csv'):
        df = pd.read_csv('results.csv')
        results = {
            'total_params': df['Total Parameters'].tolist(),
            'map50': df['Object Detection mAP50'].tolist(),
            'seg1_miou': df['Segmentation1 mIoU'].tolist(),
            'seg2_iou': df['Segmentation2 IoU'].tolist(),
            'speed': df['FPS'].tolist()
        }
        return results
    return {
        'total_params': [],
        'map50': [],
        'seg1_miou': [],
        'seg2_iou': [],
        'speed': []
    }

def validate_model(model_path, data_sets, device):
    """執行模型驗證並返回結果"""
    try:
        print(f'正在驗證模型: {model_path}')
        model = YOLO(model_path)
        
        metrics, seg, speed = model.val(
            data=data_sets,
            device=device,
            task='multi',
            name='val',
            iou=0.6,
            conf=0.001,
            classes=[2,3,4,9,10,11],
            combine_class=[2,3,4,9],
            single_cls=True,
            speed=True
        )
        
        total_params = model.info(False)[0]
        
        # 清理記憶體
        del model
        cleanup_memory()
        
        return {
            'total_params': total_params,
            'map50': metrics[0].box.map50,
            'seg1_miou': seg['seg-drivable']['mIoU'].avg,
            'seg2_iou': seg['seg-lane']['IoU'].avg,
            'speed': speed
        }
        
    except Exception as e:
        print(f"驗證過程發生錯誤: {str(e)}")
        cleanup_memory()
        return None

def save_results(results):
    """儲存結果到 CSV 檔案"""
    df = pd.DataFrame({
        'Total Parameters': results['total_params'],
        'Segmentation1 mIoU': results['seg1_miou'],
        'Segmentation2 IoU': results['seg2_iou'],
        'Object Detection mAP50': results['map50'],
        'FPS': results['speed']
    })
    df.to_csv('results.csv', index=False)

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    data_sets = 'ultralytics/datasets/bdd-multi.yaml'
    
    # 讀取已存在的結果
    results = load_existing_results()
    print(f"已讀取 {len(results['map50'])} 個已驗證的結果")
    
    # 確定從哪個迭代開始
    start_iteration = len(results['map50']) - 1  # -1 是因為包含了原始模型
    if start_iteration < 0:
        # 如果沒有任何結果，先驗證原始模型
        base_results = validate_model('ultralytics/v4.pt', data_sets, device)
        if base_results:
            for key in results:
                results[key].append(base_results[key])
            save_results(results)
            start_iteration = 0
    
    # 從上次中斷的地方繼續驗證
    for i in range(start_iteration, 12):
        print(f'正在驗證第 {i} 次迭代的模型')
        model_path = f'ultralytics/run/multi2/step_{i}_finetune2/weights/last.pt'
        
        iter_results = validate_model(model_path, data_sets, device)
        if iter_results:
            for key in results:
                results[key].append(iter_results[key])
            # 每次驗證後都儲存結果
            save_results(results)
        
        cleanup_memory()
    
    # 繪製圖表
    plot_results(results)

def plot_results(results):
    indices = list(range(len(results['map50'])))
    plt.figure(figsize=(18, 5))
    
    titles = {
        'map50': ('Object Detection Performance', 'mAP50', 'b', 'mAP'),
        'seg1_miou': ('Drivable Area Segmentation Performance', 'mIoU', 'g', 'mIoU'),
        'seg2_iou': ('Lane Segmentation Performance', 'IoU', 'r', 'IoU'),
        'total_params': ('Total Parameters', 'Parameter', 'y', 'Total Parameters'),
        'speed': ('FPS @ batch_size = 1', 'FPS', 'c', 'FPS')
    }
    
    for i, (key, (title, ylabel, color, label)) in enumerate(titles.items(), 1):
        plt.subplot(1, 5, i)
        plt.plot(indices, results[key], marker='o', linestyle='-', color=color, label=label)
        plt.xlabel('origin -> iteration')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(indices)
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('comparison_pruned.png')
    plt.close()

if __name__ == "__main__":
    main()