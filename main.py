# main.py
import cv2
from ultralytics import YOLO
from face_recognition_module import is_employee
from action_recognition_module import detect_action
from arm_detection_module import check_for_collision

# 加载YOLO模型
model = YOLO("yolov8_pose_model.pt")

# 定义监控区域与机械臂位置
arm_position = (5, 5)  # 示例机械臂位置
person_position = (6, 6)  # 示例人员位置

# 视频流处理
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)

    for result in results:
        keypoints = result.keypoints
        # 识别人员身份
        employee_id = is_employee(frame)
        
        if employee_id:
            action = detect_action(keypoints)
            print(f"Employee {employee_id} detected. Action: {action}")
        else:
            print("Non-employee detected. Alert: Please leave the area.")

        # 判断是否有碰撞风险
        if check_for_collision(arm_position, person_position):
            print("Collision risk detected! Stop the arm!")

    cv2.imshow("Security Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
