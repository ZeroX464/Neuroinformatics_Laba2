import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Загрузка данных из Excel файла
def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Создание класса многослойного персептрона
class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(FlexibleMLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) 
            for i in range(len(hidden_sizes) - 1)
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = activation
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x

# Функция обучения
def train_model(model, criterion, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            labels = labels.long()  # Приведение типа меток к Long
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")

# Загрузка данных
X_train, X_test, y_train, y_test = load_data("C:\\TEST\\Laba1\\Neuroinformatics_Laba2\\Neuroinformatics_Laba2\\plant_dataset.xlsx")

# Параметры модели и обучения
input_size = X_train.shape[1]
hidden_sizes = [9]  # Количество нейронов в каждом скрытом слое
output_size = len(set(y_train))
activation = nn.ReLU()
epochs = 100
lr = 0.001
batch_size = 10

# Создание модели
model = FlexibleMLP(input_size, hidden_sizes, output_size, activation)

# Критерий и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Создание DataLoader
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Обучение модели
train_model(model, criterion, optimizer, train_loader, epochs)

# Оценка модели на тестовых данных
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy}")
