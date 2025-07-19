import numpy as np

# Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid 函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

class YihuoBP: #2 3 1
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.normal(0, np.sqrt(2.0/input_size), (input_size, hidden_size))
        self.weights_hidden_output = np.random.normal(0, np.sqrt(2.0/hidden_size), (hidden_size, output_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
      
    def forward(self, X):
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # 通过激活函数得到隐藏层输出 激活函数改造为0-1
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)
        
        # 计算输出层加权输入 同上
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        # 通过激活函数得到最终输出 同上
        output = sigmoid(self.output_layer_activation)
        
        return output
    
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

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 反向传播
            self.backward(X, y, output)
            
def main():
    # XOR 
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [1], [1], [0]])

    yihuo = YihuoBP(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)
    
    yihuo.train(input_data, output_data, epochs=10000)

    for i in range(len(input_data)):
        prediction = yihuo.forward(input_data[i])
        print(f"input: {input_data[i]}, yuce_output: {prediction}")

if __name__ == "__main__":
    main()

    

