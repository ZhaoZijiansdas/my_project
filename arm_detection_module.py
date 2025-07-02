# arm_detection_module.py
def check_for_collision(arm_position, person_position):
    """
    检测机械臂与人员是否可能发生碰撞
    :param arm_position: 机械臂当前位置
    :param person_position: 人员当前位置
    :return: 是否存在碰撞风险
    """
    arm_x, arm_y = arm_position
    person_x, person_y = person_position
    distance = ((arm_x - person_x)**2 + (arm_y - person_y)**2)**0.5
    danger_threshold = 1.5  # 设置一个碰撞阈值，单位可以是米

    if distance < danger_threshold:
        return True  # 存在碰撞风险
    return False  # 不存在碰撞风险
