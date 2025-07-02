# action_recognition_module.py
# action_recognition_module.py

def detect_action(keypoints):
    """
    使用姿态估计关键点判断是否有危险动作
    :param keypoints: 姿态估计的关键点数据
    :return: 动作类型（危险或安全）
    """
    if len(keypoints) < 17:  # 假设我们需要至少17个关键点来判断动作
        return "No person detected"
    
    # 调试：输出每个关键点的详细信息
    for i, point in enumerate(keypoints):
        #print(f"Keypoint {i}: {point}")

        # 获取关键点位置：假设右肩、左肩、右膝、左膝
        right_shoulder = keypoints[6][1]
        left_shoulder = keypoints[5][1]
        right_knee = keypoints[9][1]
        left_knee = keypoints[8][1]
        hip = keypoints[11][1]  # 假设使用臀部来帮助判断是否蹲#

    # 判断倚靠：右肩低于左肩，且接近水平或前倾
    if right_shoulder > left_shoulder and abs(right_shoulder - left_shoulder) < 20:
        return "Dangerous behavior: Leaning"

    # 判断蹲下：膝盖低于臀部（表明身体位置较低）
    elif right_knee > hip and left_knee > hip:
        return "Dangerous behavior: Crouching"
    
    # 判断等待或静止：肩膀的高度相近，表明没有明显的动作
    elif abs(right_shoulder - left_shoulder) < 10:
        return "Dangerous behavior: Waiting"



    return "Safe"

