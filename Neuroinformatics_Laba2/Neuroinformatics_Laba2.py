import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Инициализация весов
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # Инициализация смещений
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.bias_output = np.random.randn(1, self.output_size)
        
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Прямое распространение
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.softmax(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.softmax(self.output_input)
        
        return self.predicted_output
    
    def backward(self, X, y, output, learning_rate):
        # Обратное распространение ошибки
        self.output_error = y - output
        output_delta = self.output_error
        
        self.hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = self.hidden_error * self.hidden_output * (1 - self.hidden_output)
        
        # Обновление весов и смещений
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs+1):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                mse = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, MSE: {mse}')

# Загрузка данных из файла
data = pd.read_excel("C:\\Oleg\\LearningData.xlsx", header = 0)

# Разделение данных на параметры и результат
X = data.iloc[:, :-1].values
y_text = data.iloc[:, -1].values.reshape(-1, 1)

# Перевод названий классов в числа
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text.ravel())

# Нормирование входных данных
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Инициализация параметров нейросети
input_size = X.shape[1]
hidden_size = 15
output_size = len(label_encoder.classes_)

# Преобразование классов в вектора 0 и 1
y_onehot = np.zeros((len(y_encoded), output_size))
y_onehot[np.arange(len(y_encoded)), y_encoded] = 1

# Инициализация и тренировка нейросети
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y_onehot, epochs=1000, learning_rate=0.1)
