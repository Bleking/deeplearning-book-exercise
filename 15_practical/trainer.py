from copy import deepcopy
import numpy as np

import torch


class Trainer():
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        super().__init__()  # super.__init__()가 아니다. 주의!
        
    def _batchify(self, x, y, batch_size, random_split=True):  # 데이터 셔플링 후 미니배치 만들기 (매 epoch마다 SGD 적용 목적)
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)

            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)  # 배치 크기별로 나누기
        y = y.split(batch_size, dim=0)

        return x, y
    
    def _train(self, x, y, config):
        self.model.train()

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.criterion(y_hat_i, y_i.squeeze())

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:  # 현재 학습 상황
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)
    
    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.criterion(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)
    
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for idx in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())  # 모델의 weight 파라미터 값을 json 형태로 변환해서 best_model에 복사

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                idx + 1, config.n_epochs, train_loss, valid_loss, lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)  # best_model 값을 self.model에 로딩
