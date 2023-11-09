import torch
import torch.nn

import sys
import numpy as np
import matplotlib.pyplot as plt

from mnist_classifier.utils import load_mnist
from mnist_classifier.utils import get_model

model_fn = './model.pth'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load(fn, device):
    d = torch.load(fn, map_location=device)

    return d['model'], d['config']

def plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)

        plt.imshow(img, cmap='gray')
        plt.show()
        print("Predict:", float(torch.argmax(y_hat[i], dim=-1)))

# 문제점: 미니배치 단위로 추론을 하지 않는다. 테스트셋이 한번에 연산하기 너무 클 경우 메모리 부족 오류 발생.
def test(model, x, y, to_be_shown=True):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt

        print("Accuracy: %.4f" % accuracy)

        if to_be_shown:
            plot(x, y_hat)

model_dict, train_config = load(model_fn, device)
print(train_config)

# MNIST 테스트셋 불러오기
x, y = load_mnist(is_train=False, flatten=(train_config.model == 'fc'))
x, y = x.to(device), y.to(device)
print(x.shape, y.shape)

input_size = int(x.shape[-1])
output_size = int(max(y)) + 1

model = get_model(input_size, output_size, train_config, device)

model.load_state_dict(model_dict)

test(model, x, y, to_be_shown=False)

# n_test = 20
# test(model, x[:n_test], y[:n_test], to_be_shown=True)
