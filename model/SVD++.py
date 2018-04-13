from recommender.model.mf import MF
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict

class SVDpp(MF):
    """
    implement FunkSVD
    """
    def __init__(self):
        super(SVDpp, self).__init__()
        self.config.lambdaB = 0.01  # 偏置项系数
        self.config.lambdaY = 0.01
        self.init_model()

    def init_model(self):
        super(SVDpp, self).init_model()
        self.Bu = np.random.rand(self.rg.get_train_size()[0])  # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1])  # bias value of item
        self.Y = np.random.rand(self.rg.get_train_size()[1],self.config.factor)

        self.user2sum = defaultdict(dict)
        self.user2number = defaultdict(dict)


    def train_model(self):
        for iteration in range(15):
            print(f'第{iteration}次训练')
            self.loss = 0
            for index, line in tqdm(enumerate(self.rg.train_set())):
                user, item, rating = line
                u = self.rg.train_user[user]
                i = self.rg.train_item[item]
                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                # sum_y_number, sum_y = self.get_sum(user)
                # update
                self.P[u] += self.config.lr * (error * q - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * (p + self.user2sum[user]) - self.config.lambdaQ * q)

                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * self.Bu[u])
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * self.Bi[i])

                Ui = self.rg.get_user_items(user)
                # print('here')
                for j in Ui:
                    index_j = self.rg.train_item[j]
                    self.Y[index_j] += self.config.lr * (error / math.sqrt(self.user2number[user]) * q -
                                                         self.config.lambdaY * self.Y[index_j])
            self.loss += self.config.lambdaP * (self.P * self.P).sum() +\
                         self.config.lambdaQ * (self.Q * self.Q).sum() +\
                         self.config.lambdaB * (self.Bi * self.Bi).sum() +\
                         self.config.lambdaB * (self.Bu * self.Bu).sum() +\
                        self.config.lambdaY * (self.Y * self.Y).sum()

            if self.isConverged(iteration):
                break

    def predict(self, u, i):
        number, sum_y = self.get_sum(u)
        self.user2sum[u] = sum_y
        self.user2number[u] = number
        if self.rg.contains_user(u) and self.rg.contains_item(i):
            u = self.rg.train_user[u]
            i = self.rg.train_item[i]
            return self.Q[i].dot(self.P[u]+sum_y) + self.rg.globalMean + self.Bi[i] + self.Bu[u]
        else:
            return self.rg.globalMean

    def get_sum(self, u):
        Ui = self.rg.get_user_items(u)
        sum_y = np.zeros(self.config.factor)
        for j in Ui:
            sum_y += self.Y[self.rg.train_item[j]]
        return len(Ui), sum_y / (math.sqrt(len(Ui)))

if __name__ == '__main__':
    bmf = SVDpp()
    bmf.train_model()
    bmf.show_rmse()
