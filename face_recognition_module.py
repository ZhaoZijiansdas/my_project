# face_recognition_module.py
import face_recognition
import numpy as np

# 假设已加载员工人脸数据库
employee_db = {
    "employee_id_123": "employee_image.jpg"
}

def is_employee(image):
    """
    检查图像中的人脸是否为员工
    :param image: 输入的图像数据
    :return: 如果是员工，返回员工ID，否则返回None
    """
    # 使用face_recognition获取图像中的人脸编码
    encoding = face_recognition.face_encodings(image)
    if encoding:
        # 与员工数据库中的人脸进行匹配
        for emp_id, emp_image_path in employee_db.items():
            emp_image = face_recognition.load_image_file(emp_image_path)
            emp_encoding = face_recognition.face_encodings(emp_image)
            if emp_encoding and np.allclose(encoding[0], emp_encoding[0]):
                return emp_id
    return None
