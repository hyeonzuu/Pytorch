import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 랜덤 씨드
torch.manual_seed(1)

# 변수 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# print(x_train)
# print(x_train.shape)
# print(y_train)
# print(y_train.shape)

# 가중치와 편향 초기화
# 가중치 w를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함
W = torch.zeros(1, requires_grad=True)
#print(W)

b = torch.zeros(1, requires_grad=True)
#print(b)

# 경사하강법 구현하기
optimazer = optim.SGD([W, b], lr=0.01)


nb_epochs = 1999 #원하는 만큼 경사 하강법 반복
for epoch in range(nb_epochs + 1):

    # 가설 세우기
    hypothesis = x_train * W + b
    # print(hypothesis)

    # 비용함수 선언하기
    cost = torch.mean((hypothesis - y_train) ** 2)  # 평균 구하기
    # print(cost)

    # gradient를 0으로 초기화
    optimazer.zero_grad()
    # 비용함수 미분하여 gradient 계산
    cost.backward()
    # w와 b를 업데이트
    optimazer.step()

    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f}, cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))