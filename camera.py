import cv2
import numpy as np
from BP import BP
import os
import time  # 导入time模块以便于命名文件时使用时间戳

# 加载训练好的权重
bp = BP(input_size=128*128, hidden_size=128, output_size=3, learning_rate=0.01)
bp.load_weights('C:/Users/Administrator/Desktop/BP/src/data.npz')

# 创建 VideoCapture 对象，读取摄像头视频
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 读取视频帧
while True:
    ret, frame = cap.read()
    
    # 如果读取到最后一帧，退出循环
    if not ret:
        break

    # 预处理帧
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    resized_frame = cv2.resize(gray_frame, (128, 128))
    normalized_frame = cv2.Canny(resized_frame, 50, 150)  # 添加边缘检测，提取手势轮廓
    normalized_frame = normalized_frame / 255.0
    input_data = normalized_frame.flatten()

    #.reshape(1, -1)

    # 使用BP神经网络预测
    prediction = bp.predict(input_data)
    predicted_label = np.argmax(prediction)

    # 显示预测结果
    label_map = {0: "shitou", 1: "jiandao", 2: "bu"}
     #cv2.putText(image, text, org, fontFace, fontScale, color, thickness)
    cv2.putText(frame, f"yuce: {label_map[predicted_label]} , {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('Camera', frame)
    # 添加键盘控制以保存图片
    key = cv2.waitKey(25) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f'C:/Users/Administrator/Desktop/BP/src/shitou/shitou_{int(time.time())}.jpg', frame)
        print("保存石头图片")
    elif key == ord('j'):
        cv2.imwrite(f'C:/Users/Administrator/Desktop/BP/src/jiandao/jiandao_{int(time.time())}.jpg', frame)
        print("保存剪刀图片")
    elif key == ord('b'):
        cv2.imwrite(f'C:/Users/Administrator/Desktop/BP/src/bu/bu_{int(time.time())}.jpg', frame)
        print("保存布图片")
    # 按下 'q' 键退出
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()