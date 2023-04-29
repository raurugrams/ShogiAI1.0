import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

# ニューラルネットワークの設計
class ShogiNet(nn.Module):
    def __init__(self):
        super(ShogiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 9 * 9, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1, 64 * 9 * 9)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

# 前処理されたデータを使用してモデルを学習
def train(model, train_data, val_data, epochs, batch_size, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        
        val_loss /= len(val_data)
        print(f"Epoch {epoch + 1}/{epochs}, Validation loss: {val_loss}")

def preprocess_data():
    # 棋譜データの読み込みと前処理を行う
    num_samples = 1000
    dummy_board_data = np.random.random((num_samples, 3, 9, 9))
    dummy_eval_data = np.random.random((num_samples, 1))

    # データを訓練データと検証データに分割
    split_idx = int(num_samples * 0.8)
    train_data = data.TensorDataset(torch.tensor(dummy_board_data[:split_idx], dtype=torch.float32), torch.tensor(dummy_eval_data[:split_idx], dtype=torch.float32))
    val_data = data.TensorDataset(torch.tensor(dummy_board_data[split_idx:], dtype=torch.float32), torch.tensor(dummy_eval_data[split_idx:], dtype=torch.float32))

    return train_data, val_data

# データの前処理・学習の実行
train_data, val_data = preprocess_data()
model = ShogiNet()
train(model, train_data, val_data, epochs=50, batch_size=32, learning_rate=1e-3)