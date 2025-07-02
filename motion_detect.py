import cv2
import logging
from ultralytics import YOLO
from action_recognition_module import detect_action

# 禁用YOLO日志
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# 加载YOLO模型
model = YOLO("F:/paper code/dangerous_detect/yolov8n-pose.pt")

# 视频流处理
cap = cv2.VideoCapture(0)  # 使用默认摄像头

while True:
    ret, frame = cap.read()

    # 使用YOLOv8进行姿态估计（关键点检测）
    results = model(frame)

    for result in results:
      keypoints = result.keypoints
    
    if keypoints is not None and len(keypoints.xy) > 0:
        person_kpts = keypoints.xy[0].cpu().numpy().tolist()  # 提取关键点坐标
        action = detect_action(person_kpts)
    else:
        action = "No person detected"

    print(f"Action: {action}")
    cv2.putText(frame, f"Action: {action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # # 显示结果
    # cv2.imshow("Action Recognition", frame)

    # # 按'q'退出
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
  # 显示视频帧
    cv2.imshow('Video', result.plot())  # 使用YOLO结果中的plot方法来显示关键点标注的视频

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
