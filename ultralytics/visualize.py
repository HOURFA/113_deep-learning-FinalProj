import cv2
import numpy as np

color_dict = {0: (0, 0, 255), 
              1: (0, 255, 0), 
              2: (255, 0, 0), 
              3: (255, 255, 0), 
              4: (0, 255, 255), 
              5: (255, 0, 255), 
              6: (128, 0, 0), 
              7: (0, 128, 0), 
              8: (0, 0, 128), 
              9: (128, 128, 0)}  # Define colors for different classes
cls_dict = {0: 'vehicle', 
            1: 'rider', 
            2: 'car', 
            3: 'bus', 
            4: 'truck', 
            5: 'bike', 
            6: 'motor', 
            7: 'traffic light', 
            8: 'traffic sign', 
            9: 'train'}  # Define class names for different classes

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draws a border around a region of interest in an image.

    Args:
        img (numpy.ndarray): The input image.
        top_left (tuple): The coordinates of the top-left corner of the region of interest.
        bottom_right (tuple): The coordinates of the bottom-right corner of the region of interest.
        color (tuple, optional): The color of the border in BGR format. Defaults to (0, 255, 0) (green).
        thickness (int, optional): The thickness of the border lines. Defaults to 10.
        line_length_x (int, optional): The length of the horizontal lines extending from the top-left and top-right corners. Defaults to 200.
        line_length_y (int, optional): The length of the vertical lines extending from the top-left and bottom-left corners. Defaults to 200.

    Returns:
        numpy.ndarray: The image with the border drawn.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


#draww road and lane on frame
def draw_image(frame, box_list, road, lane, draw_lane, draw_road, draw_box, transparency_factor):
    """
    Draws the road, lane, and bounding boxes on the input frame.

    Args:
        frame (numpy.ndarray): The input frame to draw on.
        box_list (list): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        road (numpy.ndarray): The road mask.
        lane (numpy.ndarray): The lane mask.
        draw_lane (bool): Whether to draw the lane on the frame.
        draw_road (bool): Whether to draw the road on the frame.
        draw_box (bool): Whether to draw the bounding boxes on the frame.
        transparency_factor (float): The transparency factor for blending the overlays.

    Returns:
        numpy.ndarray: The frame with the overlays drawn.
    """
    frame_b,frame_g,frame_r = cv2.split(frame)
    if draw_road:
        color_road = np.stack([frame_b, road * 255, frame_r], axis=-1)        
        frame = cv2.addWeighted(frame, 1, color_road, transparency_factor, 0)
    if draw_lane:
        color_lane = np.stack([lane * 255, frame_g, frame_r], axis=-1)
        frame = cv2.addWeighted(frame, 1, color_lane, transparency_factor, 0)
    if draw_box and box_list is not None:        
        for box in box_list:
            if len(box) == 7:
                x_min, y_min, x_max, y_max, cls, cof, lane_num = box
                lane_idx = -1
            else:
                x_min, y_min, x_max, y_max, cls, cof = box
                lane_num = 0
                lane_idx = -1
            draw_border(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color_dict[cls], thickness=2, line_length_x=30, line_length_y=30)
            cv2.putText(frame, cls_dict[cls], (int(x_min), int(y_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_dict[cls], 2)
            cv2.rectangle(frame, (int(x_min), int(y_min-30)), (int(x_min)+120, int(y_min)), color_dict[cls], -1)
            if lane_idx >= 0 :
                cv2.putText(frame, f"{cls_dict[cls]},{lane_idx:.0f}", (int(x_min)+5, int(y_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            else:                
                cv2.putText(frame, f"{cls_dict[cls]}, {lane_num:.0f}", (int(x_min)+5, int(y_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    return frame

def put_fps(frame, recognition_time, avg_fps):
    """
    Add FPS information to the given frame.

    Args:
        frame (numpy.ndarray): The frame to add the FPS information to.
        recognition_time (float): The time taken for recognition in seconds.
        avg_fps (float): The average frames per second.
    Returns:
        None
    """
    cv2.putText(frame, 'FPS:' + str(recognition_time), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'AVG_FPS:' + str(avg_fps), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 