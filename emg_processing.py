import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F


# CSV 파일 읽기
data = pd.read_csv('emg.csv')

# 특성 및 레이블 분리
X = data.iloc[:, :-1].values  # 마지막 열을 제외한 모든 열을 특성으로 간주
y = data.iloc[:, -1].values   # 마지막 열을 레이블로 간주

# train, test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 레이블 인덱스 0부터 시작하도록 조정
y_train = y_train - 1
y_test = y_test - 1

# 고유한 클래스 수 계산 (레이블 조정 후)
num_classes = len(np.unique(y_train))

# 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 파이토치 텐서로 변환
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataset과 DataLoader 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 신경망 모델 정의
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (input_size // 4), 1000)  # 입력 크기에 맞게 수정 필요
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))  # 입력을 4D 텐서로 변환하여 합성곱 층에 전달
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (input_size // 4))  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화
input_size = X_train.shape[1]
model = CNN(input_size, num_classes)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 모델 학습 실행
train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# 모델 평가
def evaluate_model(model, test_loader):
    model.eval()  # 평가 모드로 설정
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# 테스트 세트에서 모델 평가
accuracy = evaluate_model(model, test_loader)
print(f'Accuracy: {accuracy:.4f}')
