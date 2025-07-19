import numpy as np
import cv2
import os

# Sigmoid 激活函数
def sigmoid(x):
    # 防止溢出
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Sigmoid 函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

# Softmax 激活函数（用于多分类输出层）
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# BP 神经网络类
class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化BP神经网络
        input_size: 输入层神经元数量
        hidden_size: 隐藏层神经元数量
        output_size: 输出层神经元数量
        learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 使用Xavier初始化权重
        self.weights_input_hidden = np.random.normal(0, np.sqrt(2.0/input_size), 
                                                   (input_size, hidden_size))
        self.weights_hidden_output = np.random.normal(0, np.sqrt(2.0/hidden_size), 
                                                    (hidden_size, output_size))
        
        # 初始化偏置为零
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
        # 记录训练过程中的损失
        self.losses = []

    def forward(self, X):
        """前向传播"""
        # 输入层到隐藏层
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = sigmoid(self.z1)
        
        # 隐藏层到输出层
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = softmax(self.z2)
        
        return self.a2

    def backward(self, X, y, output):
        """反向传播"""
        m = X.shape[0]  # 样本数量
        
        # 输出层误差
        dz2 = output - y
        dw2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # 隐藏层误差
        dz1 = np.dot(dz2, self.weights_hidden_output.T) * sigmoid_derivative(self.a1)
        dw1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # 更新权重和偏置
        self.weights_hidden_output -= self.learning_rate * dw2
        self.bias_output -= self.learning_rate * db2
        self.weights_input_hidden -= self.learning_rate * dw1
        self.bias_hidden -= self.learning_rate * db1

    def compute_loss(self, y_true, y_pred):
        """计算交叉熵损失"""
        m = y_true.shape[0]
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def train(self, X, y, epochs=1000, verbose=True):
        """训练神经网络"""
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y, output)
            self.losses.append(loss)
            
            # 反向传播
            self.backward(X, y, output)
            
            # 打印训练进度
            if verbose and epoch % 100 == 0:
                accuracy = self.evaluate(X, y)
                print(f'Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

    def predict(self, X):
        """预测"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        return self.forward(X)

    def evaluate(self, X, y):
        """评估准确率"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

# 图片预处理函数
def preprocess_image(image_path, img_size=(64, 64)):
    """
    预处理图片
    image_path: 图片路径
    img_size: 图片大小
    """
    try:
        # 读取图片（灰度）
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return None
            
        # 调整大小
        img = cv2.resize(img, img_size)
        
        # 归一化到0-1范围
        img = img.astype(np.float32) / 255.0
        
        # 展平为一维向量
        img_flattened = img.flatten()
        
        return img_flattened
    except Exception as e:
        print(f"处理图片时出错 {image_path}: {e}")
        return None

# 数据加载函数
def load_dataset():
    """
    加载石头剪刀布数据集
    """
    # 定义类别和对应的文件夹
    categories = {
        'shitou': 0,    # 石头
        'jiandao': 1,   # 剪刀
        'bu': 2         # 布
    }
    
    X = []
    y = []
    
    # 获取当前脚本所在目录
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    for category, label in categories.items():
        folder_path = os.path.join(base_path, category)
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在")
            continue
            
        print(f"正在加载 {category} 类别的图片...")
        image_count = 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(folder_path, filename)
                img_features = preprocess_image(image_path)
                
                if img_features is not None:
                    X.append(img_features)
                    y.append(label)
                    image_count += 1
        
        print(f"{category} 类别加载了 {image_count} 张图片")
    
    if len(X) == 0:
        raise ValueError("没有找到任何有效的图片文件！")
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    # 将标签转换为one-hot编码
    y_onehot = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        y_onehot[i, label] = 1
    
    return X, y_onehot, y

# 数据划分函数
def train_test_split_simple(X, y, test_size=0.2, random_state=42):
    """简单的训练测试集划分"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # 随机打乱索引
    indices = np.random.permutation(n_samples)
    
    # 划分索引
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # 返回划分后的数据
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 测试单张图片
def test_single_image(model, image_path):
    """测试单张图片"""
    categories = ['石头', '剪刀', '布']
    
    # 预处理图片
    img_features = preprocess_image(image_path)
    if img_features is None:
        print("无法处理图片")
        return
    
    # 预测
    img_features = img_features.reshape(1, -1)  # 添加batch维度
    probabilities = model.predict_proba(img_features)[0]
    prediction = model.predict(img_features)[0]
    
    print(f"图片: {image_path}")
    print(f"预测结果: {categories[prediction]}")
    print("各类别概率:")
    for i, category in enumerate(categories):
        print(f"  {category}: {probabilities[i]:.4f}")
    print()

def main():
    """主函数"""
    print("=== 石头剪刀布BP神经网络识别系统 ===\n")
    
    try:
        # 1. 加载数据集
        print("1. 加载数据集...")
        X, y_onehot, y_labels = load_dataset()
        print(f"数据集大小: {X.shape[0]} 张图片")
        print(f"图片特征维度: {X.shape[1]}")
        print()
        
        # 2. 划分训练集和测试集
        print("2. 划分训练集和测试集...")
        X_train, X_test, y_train, y_test = train_test_split_simple(
            X, y_onehot, test_size=0.2, random_state=42
        )
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        print()
        
        # 3. 创建和训练神经网络
        print("3. 创建BP神经网络...")
        # 输入层: 图片像素数量 (64*64=4096)
        # 隐藏层: 128个神经元
        # 输出层: 3个神经元（石头、剪刀、布）
        model = BPNeuralNetwork(
            input_size=X.shape[1], 
            hidden_size=128, 
            output_size=3, 
            learning_rate=0.01
        )
        print()
        
        print("4. 开始训练...")
        model.train(X_train, y_train, epochs=500, verbose=True)
        print("训练完成！\n")
        
        # 4. 评估模型
        print("5. 评估模型性能...")
        train_accuracy = model.evaluate(X_train, y_train)
        test_accuracy = model.evaluate(X_test, y_test)
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print()
        
        # 5. 测试具体图片
        print("6. 测试具体图片...")
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 测试每个类别的第一张图片
        test_images = [
            os.path.join(base_path, 'shitou', 'shitou1.jpg'),
            os.path.join(base_path, 'jiandao', 'jiandao1.jpg'),
            os.path.join(base_path, 'bu', 'bu1.jpg')
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_single_image(model, img_path)
        
        return model
        
    except Exception as e:
        print(f"运行时出错: {e}")
        return None

if __name__ == "__main__":
    model = main()
