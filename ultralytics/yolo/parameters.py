import tensorrt as trt
import numpy as np
import ctypes

def inspect_engine(engine_path):
    """
    檢查TensorRT engine中的參數資料型態
    
    Args:
        engine_path: .engine文件的路徑
    """
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 載入引擎
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    print("模型資訊：")
    print("-" * 50)
    
    # 檢查每一層的資訊
    for idx in range(engine.num_layers):
        layer = engine.get_layer(idx)
        print(f"\n層 {idx}: {layer.name}")
        
        # 輸入張量資訊
        for i in range(layer.num_inputs):
            tensor = layer.get_input(i)
            print(f"輸入 {i}:")
            print(f"  名稱: {tensor.name}")
            print(f"  形狀: {tensor.shape}")
            print(f"  資料型態: {tensor.dtype}")
            
        # 輸出張量資訊
        for i in range(layer.num_outputs):
            tensor = layer.get_output(i)
            print(f"輸出 {i}:")
            print(f"  名稱: {tensor.name}")
            print(f"  形狀: {tensor.shape}")
            print(f"  資料型態: {tensor.dtype}")
            
    # 顯示整體模型的輸入輸出資訊
    print("\n整體模型資訊：")
    print("-" * 50)
    print("\n輸入張量：")
    for binding in engine:
        if engine.binding_is_input(binding):
            print(f"名稱: {binding}")
            shape = engine.get_binding_shape(binding)
            dtype = engine.get_binding_dtype(binding)
            print(f"形狀: {shape}")
            print(f"資料型態: {dtype}")
    
    print("\n輸出張量：")
    for binding in engine:
        if not engine.binding_is_input(binding):
            print(f"名稱: {binding}")
            shape = engine.get_binding_shape(binding)
            dtype = engine.get_binding_dtype(binding)
            print(f"形狀: {shape}")
            print(f"資料型態: {dtype}")

# 使用方式
if __name__ == "__main__":
    engine_path = "./ultralytics/pruned.engine"
    inspect_engine(engine_path)