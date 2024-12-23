import os

# 輸入資料夾路徑
input_folder = r"C:\Users\abc78\YOLOv8-multi-task-main\ultralytics\datasets\images\valid\labels"
output_folder = r"C:\Users\abc78\YOLOv8-multi-task-main\ultralytics\datasets\images\valid"

detection_object_folder = os.path.join(output_folder, "detection-object\labels")  # 如果開頭為0、2、3、4、6、7、8
seg_drivable_folder = os.path.join(output_folder, "seg-drivable\labels")  # 如果開頭為5
seg_lane_folder = os.path.join(output_folder, "seg-lane\labels")  # 如果開頭為1
os.makedirs(detection_object_folder, exist_ok=True)
os.makedirs(seg_drivable_folder, exist_ok=True)
os.makedirs(seg_lane_folder, exist_ok=True)

# 獲取資料夾中的所有txt文件
txt_files = [file for file in os.listdir(input_folder) if file.endswith(".txt")]

for txt_file in txt_files:
    input_file_path = os.path.join(input_folder, txt_file)

    with open(input_file_path, "r") as input_file:
        for line in input_file:
            # 提取每一行的第一個字元
            first_char = line[0]

            # 重新排序開頭字元
            if first_char == '0':
                new_first_char = '0';
            elif first_char == '2':
                new_first_char = '1'
            elif first_char == '3':
                new_first_char = '2'
            elif first_char == '4':
                new_first_char = '3'
            elif first_char == '6':
                new_first_char = '4'
            elif first_char == '7':
                new_first_char = '5'
            elif first_char == '8':
                new_first_char = '6'
            elif first_char == '5':
                new_first_char = '7'
            elif first_char == '1':
                new_first_char = '8'
            else:
                continue

            # 打開相應的目標文件夾
            if first_char in ('0', '2', '3', '4', '6', '7', '8'):
                output_folder = detection_object_folder
            elif first_char == '5':
                output_folder = seg_drivable_folder
            elif first_char == '1':
                output_folder = seg_lane_folder
            else:
                # 如果需要處理其他情況，可以添加相應的分支
                continue

            # 確定目標文件路徑
            output_file_path = os.path.join(output_folder, txt_file)

            # 寫入文件
            with open(output_file_path, "a") as output_file:
                # 將開頭字元重新排序後寫入文件
                output_file.write(new_first_char + line[1:])

# 完成轉換
print("文件轉換完成。")
