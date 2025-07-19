import cv2
import numpy as np
from BP import BP
import os

# 处理图片，将其转换为适合神经网络输入的格式
def imagetransfer(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (128, 128))
    img = cv2.Canny(img, 50, 150)  # 添加边缘检测，提取手势轮廓
    #cv2.imshow('Processed Image', img)  # 显示处理后的图片
    img = img / 255.0  # 归一
    """     
    cv2.imshow('Processed Image', img)  # 显示处理后的图片
    cv2.waitKey(0)  # 等待用户按键关闭窗口
    """
    return img.flatten()  # 拉平成一维向量

# 从文件夹中获取图片路径
def get_image_paths(folder):
    image_paths = []
    for filename in os.listdir(folder):
        image_paths.append(os.path.join(folder, filename))
    return image_paths

# 准备数据集
def dataprepare(folder_paths, labels): 
    X = []
    y = []
    # 遍历每个文件夹路径和对应的标签
    for folder, label in zip(folder_paths, labels):
        image_paths = get_image_paths(folder)
        for path in image_paths:
            X.append(imagetransfer(path))
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 定义文件夹路径和对应标签
    folder_paths = ['src/shitou', 'src/jiandao', 'src/bu']
    labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 编码标签

    # 准备数据集
    X, y = dataprepare(folder_paths, labels)

    # 创建BP神经网络实例
    bp = BP(input_size=X.shape[1], hidden_size=128, output_size=3, learning_rate=0.01)
    bp.train(X, y, epochs=2000)

    # 保存训练好的权重和偏置到文件
    bp.save_weights('src/data')

    # 预测
    pred = bp.predict(X)
    #print("预测结果：", np.argmax(pred))
    print("预测结果：", pred)


