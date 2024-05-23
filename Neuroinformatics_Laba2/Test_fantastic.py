from tkinter import HIDDEN
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Загрузка данных из Excel файла
def load_data(file_path):
    df = pd.read_excel(file_path)
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values
    
    label_encoder = LabelEncoder()
    y_test = label_encoder.fit_transform(y_test)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
    
    return X_test, y_test

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

X_test, y_test = load_data("C:\\Users\\user\\Source\\Repos\\ZeroX464\\Neuroinformatics_Laba2\\Neuroinformatics_Laba2\\plant_dataset_fantastic.xlsx")

input_size = X_test.shape[1]
hidden_sizes = [20]
output_size = len(set(y_test))
activation = nn.ReLU()
epochs = 100
lr = 0.001
batch_size = 5

model = FlexibleMLP(8, hidden_sizes, 10, activation)
model.load_state_dict(torch.load("MLP.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=1)
model.eval()

classes = ["Meadow Grasses", "Desert Plants", "Aquatic Plants", "Tropical Palms", "Mountain Flowers", "Succulents", "Exotic Orchids", "Alpine Shrubs", "Fruit Trees", "Flowering Shrubs"]

with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            for i in range(len(classes)):
                if predicted[0] == i:
                    print(classes[i])
            