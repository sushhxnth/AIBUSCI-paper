import pandas as pd
import numpy as np
import torch 
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
data = pd.read_csv(r"archive (1)\exoTrain.csv")

X = data.drop('LABEL', axis=1)  # anni flux coloumns
y = data['LABEL'] # idhi target

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, 3)  # 3 classes
        )

    def forward(self, x):
        return self.model(x)

input = X_train.shape[1] #intialize model here
model = MLP(input)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 20

for epoch in range(epochs): #training loop idhi
    
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    _, preds = torch.max(val_outputs, 1)
accuracy = (preds == y_val).float().mean()

print("Validation Accuracy:", accuracy.item())