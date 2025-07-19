import numpy as np

# Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sigmoid 函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

# Softmax 激活函数（用于多分类输出层）
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# BP 神经网络类 y=wx+b
class BP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # 输入层、隐藏层、输出层的神经元数量
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        #self.losses = []

        self.weights_input_hidden = np.random.normal(0, np.sqrt(2.0/input_size), (input_size, hidden_size))
        self.weights_hidden_output = np.random.normal(0, np.sqrt(2.0/hidden_size), (hidden_size, output_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
      
  
        """
        # 记录训练过程中的损失
        # 初始化输入层到隐藏层的权重（input_size x hidden_size）w
        self.weights_input_hidden = np.random.normal(0,1,(self.input_size, self.hidden_size))
        # 初始化隐藏层到输出层的权重（hidden_size x output_size）w
        self.weights_hidden_output = np.random.normal(0,1,(self.hidden_size, self.output_size))
        
        # 初始化隐藏层和输出层的偏置 b
        self.bias_hidden = np.zeros((1,self.hidden_size))
        self.bias_output = np.zeros((1,self.output_size))
        
        
        """

    # 前向传播
    def forward(self, X):
        # 计算隐藏层加权输入 矩阵乘法在加上误差
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # 通过激活函数得到隐藏层输出 激活函数改造为0-1
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)
        
        # 计算输出层加权输入 同上
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        # 通过激活函数得到最终输出 同上
        output = softmax(self.output_layer_activation)
        
        return output


    # 反向传播与权重更新
    """
    def backward(self, X, y, output):
         m = X.shape[0]  # 样本数量
        # 计算输出层误差    真实值 - 预测值
        output_error = y - output 
        # 输出层误差乘以激活函数导数，得到输出层的梯度 
        # output_delta = output_error * sigmoid_derivative(output)

        # 计算隐藏层误差 w2
        hidden_error = (1/m) * np.dot(output_delta, self.weights_hidden_output.T)

        # 隐藏层误差乘以激活函数导数，得到隐藏层的梯度
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)

        
        # 更新隐藏层到输出层的权重和偏置
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta)
        self.bias_output += np.sum(output_delta, axis=0)

        # 更新输入层到隐藏层的权重和偏置
        self.weights_input_hidden += np.dot(X.T, hidden_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0)
    """
    def backward(self, X, y, output):
        
        m = X.shape[0]  # 样本数量
        
        # 输出层误差
        dz2 = output - y
        dw2 = (1/m) * np.dot(self.hidden_layer_output.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # 隐藏层误差
        dz1 = np.dot(dz2, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_layer_output)
        dw1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # 更新权重和偏置
        self.weights_hidden_output -= self.learning_rate * dw2
        self.bias_output -= self.learning_rate * db2
        self.weights_input_hidden -= self.learning_rate * dw1
        self.bias_hidden -= self.learning_rate * db1

    """
    def compute_loss(self, y_true, y_pred):
        #计算交叉熵损失
        m = y_true.shape[0]
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    """

    # 训练神经网络
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # 前向传播得到输出
            output = self.forward(X)

             #计算损失
            #loss = self.compute_loss(y, output)
            #self.losses.append(loss)

            #反向传播并更新参数
            self.backward(X, y, output)


    # 预测函数
    def predict(self, X):
        return self.forward(X)
    
    # 保存权重和偏置到文件
    def save_weights(self, file_path):
        np.savez(file_path,
                 weights_input_hidden=self.weights_input_hidden,
                 weights_hidden_output=self.weights_hidden_output,
                 bias_hidden=self.bias_hidden,
                 bias_output=self.bias_output)
        print(f"保存到了{file_path}.npz")
    
    # 从文件加载权重和偏置
    def load_weights(self, file_path):
        data = np.load(file_path)
        self.weights_input_hidden = data['weights_input_hidden']
        self.weights_hidden_output = data['weights_hidden_output']
        self.bias_hidden = data['bias_hidden']
        self.bias_output = data['bias_output']
