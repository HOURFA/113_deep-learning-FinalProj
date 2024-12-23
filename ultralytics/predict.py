import sys
sys.path.insert(0, "/home/rfa/NTUST/LAB/A-Yolom")
import cv2
import numpy as np
import opencv_with_cuda_api as cv2_cuda
import time
import visualize
from ultralytics.nn.modules.block import C2f_v2
def gstreamer_pipeline(
    capture_width=960,
    capture_height=540,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):    
    return (
        "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink drop=1"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def preprocess(image, th, use_cuda=False):
    """
    Preprocesses an image by applying bilateral filtering, Gaussian blur, and Canny edge detection.

    Args:
        image (numpy.ndarray): The input image.
        th (int): The threshold value for Canny edge detection.
        use_cuda (bool, optional): Whether to use CUDA for GPU acceleration. Defaults to False.

    Returns:
        numpy.ndarray: The preprocessed image with edges detected.

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    if use_cuda:
        bilateral = cv2_cuda.bilateralFilter_cuda(gray, 3, 15, 15)
        blur = cv2_cuda.gaussianblur_cuda(bilateral, (3, 3), 2)
        edge = cv2_cuda.canny_cuda(blur, 3, 20, th)
    else:
        bilateral = cv2.bilateralFilter(gray, 3, 15, 15)
        blur = cv2.GaussianBlur(bilateral, (3, 3), 2)
        edge = cv2.Canny(blur, 20, th)
        # cv2.imwrite('gray.jpg',gray)
        # cv2.imwrite('bilateral.jpg', bilateral)
        # cv2.imwrite('blur.jpg', blur)
        # cv2.imwrite('edge.jpg', edge)    
    return edge
def tensor2ndarray(tensor):
    import torch
    """
    Convert a PyTorch tensor to a NumPy ndarray.

    Args:
        tensor (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: The converted NumPy ndarray.
    """
    array = tensor[0].to(torch.uint8).cpu().numpy()

    return array
def volov8_multi_task(frame, model, TensorRt, multi):
    """
    Perform multi-task object detection and lane detection on a given frame.

    Args:
        frame (numpy.ndarray): The input frame for detection.
        TensorRt (bool): Flag indicating whether to use TensorRT for inference.

    Returns:
        tuple: A tuple containing the following elements:
            - frame (numpy.ndarray): The processed frame with detections and lane markings.
            - boxes_list (list): A list of bounding box coordinates for detected objects.
            - array_road (numpy.ndarray): The lane markings extracted from the frame.
            - array_lane (numpy.ndarray): The road region extracted from the frame.
            - process_FPS (float): The processing frames per second.

    """
    boxes_list, array_road, array_lane= [], [], []
    if multi:
        boxes_list, ploted_img = model.predict(source=frame,
                                            imgsz=(640, 640),
                                            device= 0,
                                            batch = 2,
                                            name='test',
                                            save=False,
                                            conf=0.25,
                                            iou=0.45,
                                            show_labels=True,
                                            boxes=True,
                                            show=True,
                                            TensorRt = TensorRt,
                                            task = 'multi')
        array_road = tensor2ndarray(ploted_img[1])
        array_lane = tensor2ndarray(ploted_img[2])        
    else:
        ploted_img = model.predict(source=frame,
                                    imgsz=(640, 640),
                                    device= 0,
                                    batch = 2,
                                    name='test',
                                    save=False,
                                    conf=0.25,
                                    iou=0.45,
                                    show_labels=True,
                                    boxes=True,
                                    show=True,
                                    TensorRt = TensorRt)

    return frame, boxes_list, array_road, array_lane

def postprocess(frame, box_list, road, lane):

    
    lane_frame = np.zeros_like(frame)
    
    lane_list = extend_lane(lane*255)
    lane_list.sort(key=lambda line: line[0][0])
    for line in lane_list:
        cv2.line(frame, line[0], line[1], (255, 0, 0), 2)
        cv2.line(lane_frame, line[0], line[1], (255, 0, 0), 2)
    for i, box in enumerate(box_list):
        x_min, y_min, x_max, y_max, label, confidence = box
        box_center_x = (x_min + x_max) / 2
        box__y = y_max
        cv2.rectangle(lane_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255), 2)
        min_distances = [float('inf'), float('inf')]
        closest_lanes = [-1, -1]

        for j, line in enumerate(lane_list):
            distance = horizontal_distance_to_line(line[0][0], line[0][1], line[1][0], line[1][1], box_center_x, box__y)
            if j == 0:
                min_distances[0] = distance
                closest_lanes[0] = j
            elif j == 1:
                min_distances[1] = distance
                closest_lanes[1] = j
            elif distance < min_distances[0]:
                min_distances[0] = min_distances[1]
                closest_lanes[0] = closest_lanes[1]
                min_distances[1] = distance
                closest_lanes[1] = j
        cv2.putText(lane_frame, f"{closest_lanes[1]}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        box_list[i] = (x_min, y_min, x_max, y_max, label, confidence, closest_lanes[1])

    # contours, hierarchy = cv2.findContours(open_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for i, contour in enumerate(contours):
    #     color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    #     cv2.drawContours(road_frame, [contour], -1, color, 3)


    cv2.imshow("Lane_frame", lane_frame)
    # cv2.waitKey(0)
def horizontal_distance_to_line(x1, y1, x2, y2, px, py):
    # 水平距離計算（y軸固定）
    if y1 == y2:
        return abs(px - x1)  # 水平線

    # 直線方程的斜率和截距
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    intercept = y1 - slope * x1

    # 計算同一水平上的x坐標
    x_on_line = (py - intercept) / slope if slope != float('inf') else x1
    horizontal_distance = abs(px - x_on_line)
    return horizontal_distance
def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return [int(px), int(py)]

def extend_line_to_boundaries(line, focus_point, img_height, img_width):
    x1, y1, x2, y2 = line
    # Calculate the slope
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = float('inf')

    # Extend to bottom boundary
    if slope != float('inf'):
        y_bottom = img_height - 1
        x_bottom = int(focus_point[0] - (focus_point[1] - y_bottom) / slope)
    else:
        x_bottom = x1
        y_bottom = img_height - 1

    return (x_bottom, y_bottom)

def extend_lane(lane_image):
    
    edges = cv2.Canny(lane_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)    
    lane_list = []
    unique_lines = []
    tolerance = 0.1  # 容差值，用於判斷斜率是否相似
    if lines is None:
        return lane_list
    for line in lines:
        x1, y1, x2, y2 = line[0]        
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        
        # 過濾掉近水平線的直線
        if abs(slope) > 0.1:
            # 檢查是否有相似斜率的直線
            similar_line_found = False
            for unique_line in unique_lines:
                x1_u, y1_u, x2_u, y2_u = unique_line
                slope_u = (y2_u - y1_u) / (x2_u - x1_u) if x2_u != x1_u else float('inf')
                if abs(slope - slope_u) < tolerance:
                    similar_line_found = True
                    break
            # 如果沒有相似斜率的直線，則添加到 unique_lines
            if not similar_line_found:
                unique_lines.append(line[0])

    # 找到所有直線的交點（焦點）
    intersections = []
    for i in range(len(unique_lines)):
        for j in range(i + 1, len(unique_lines)):
            intersection = find_intersection(unique_lines[i], unique_lines[j])
            if intersection:
                intersections.append(intersection)

    # 計算平均交點作為焦點
    focus_point = np.mean(intersections, axis=0).astype(int) if intersections else [lane_image.shape[1] // 2, 0]


    # 繪製唯一的直線並延伸到焦點和邊界
    img_height, img_width = lane_image.shape
    for line in unique_lines:
        bottom_point = extend_line_to_boundaries(line, focus_point, img_height, img_width)
        lane_list.append([bottom_point, (focus_point[0], focus_point[1])])

    return lane_list

def is_vertical(contour):
    x_coords = [point[0][0] for point in contour]
    y_coords = [point[0][1] for point in contour]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    return y_range > 20
def video_test(path, mode, multi):
    """
    Perform object tracking on a video file.

    Args:
        path (str): The path to the video file.
        TensorRt (bool): Flag indicating whether to use TensorRT for inference.

    Returns:
        None
    """
    if mode:
        cap = cv2.VideoCapture(path)
    else:
        pipeline = gstreamer_pipeline(capture_width=3280, capture_height=1848, 
                                display_width=1920, display_height=1080, framerate=28, flip_method=2)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)    
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    w, h  = 1280, 720
    result = cv2.VideoWriter("results.mp4",cv2.VideoWriter_fourcc(*'mp4v'),fps,(w, h))
    fps_list = []
    from ultralytics import YOLO
    model = YOLO(model_path)
    tensorRt = True if model_path.endswith(".engine" )or model_path.endswith(".onnx") else False
    if cap.isOpened():
        try:
            while True:
                success, frame = cap.read()
                
                frame = cv2.resize(frame, (w,h))
                if success:

                    start_time = time.time()
                    predict_result, boxes_list, array_road, array_lane = volov8_multi_task(frame, model, tensorRt, multi)                            
                    # postprocess(frame, boxes_list, array_road, array_lane)
                    end_time = time.time()
                    if multi:
                        predict_result = visualize.draw_image(predict_result, 
                                                              boxes_list, 
                                                              array_road, 
                                                              array_lane,
                                                              draw_lane=True,
                                                              draw_box=True,
                                                              draw_road=True,
                                                              transparency_factor=0.2)
                                        
                    recognition_time = round((1 / (end_time - start_time)), 2)
                    fps_list.append(recognition_time)
                    avg_fps = round((sum(fps_list) / len(fps_list)), 2)

                    visualize.put_fps(predict_result, recognition_time, avg_fps)

                    cv2.imshow("Predict_result", predict_result)
                    # cv2.waitKey(0)
                    result.write(predict_result)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    elif cv2.waitKey(1) & 0xFF == ord("p"):
                        cv2.waitKey(0)
                else:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    result.release()
if __name__ == "__main__":
    # model_path = "/home/rfa/NTUST/LAB/A-Yolom/ultralytics/v4.pt"
    # model_path = "/home/rfa/NTUST/LAB/A-Yolom/ultralytics/best_multi.engine"
    # model_path = "/home/rfa/NTUST/LAB/A-Yolom/ultralytics/pruned.pt"
    model_path = "/home/rfa/NTUST/LAB/A-Yolom/ultralytics/prune.engine"
    source_path = "/home/rfa/NTUST/LAB/A-Yolom/ultralytics/test/003.mp4"
    multi = True
    mode = 1 # 0: camera, 1: video
    video_test(source_path, mode, multi)
