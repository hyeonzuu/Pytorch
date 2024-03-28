import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

print(x_train.shape)
print(y_train.shape)

# 가중치와 편향 선언
W = torch.zeros((3, 1), requires_grad=True) # 곱셈 우측&행렬릐 열의 크기와 우측에 있는 행렬의 행의 크기가 일치
b = torch.zeros(1, requires_grad=True)

#optimizer 설정
optimizer = optim.SGD([W, b], lr = 1e-5)

nb_epoch = 20
for epoch in range(nb_epoch + 1):
    #h(x) 계산
    # 편향은 브로드캐스팅되어 각 샘플에 더해집니다
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    #cost로 h(x)계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6}'.format(
        epoch, nb_epoch, hypothesis.squeeze().detach(), cost.item()
    ))
#자동 미분에서 추적하는 기록을 중단