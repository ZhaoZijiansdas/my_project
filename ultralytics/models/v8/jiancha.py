import os
from glob import glob

label_dir = "G:/yolov8/yolov8_segment_pose-main/datasets/mydata/labels/train"
for path in glob(os.path.join(label_dir, "*.txt")):
    if os.path.getsize(path) == 0:
        print("❌ 空标签文件：", path)
