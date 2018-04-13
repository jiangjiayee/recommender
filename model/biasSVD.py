from recommender.model.mf import MF
import numpy as np
from tqdm import tqdm

class biasSVD(MF):
    """
    implement biasSVD
    """

    def __init__(self):
        super(biasSVD, self).__init__()
        self.config.lambdaB = 0.001  # 偏置项系数
        self.init_model()

        self.Bu = np.random.rand(self.rg.get_train_size()[0])    # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1])    # bias value of item

    def train_model(self):
        for iteration in range(self.config.maxIteration):
            self.loss = 0
            for index, line in tqdm(enumerate(self.rg.train_set())):

                user, item, rating = line
                u = self.rg.train_user[user]
                i = self.rg.train_item[item]
                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                #updata
                self.P[u] += self.config.lr * (error * q - self.config.lambdaP * self.P[u])
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * self.Q[i])

                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * self.Bu[u])
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * self.Bi[i])

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum()\
                        + self.config.lambdaB * ((self.Bu * self.Bu).sum()+(self.Bi * self.Bi).sum())
            if self.isConverged(iteration):
                break



    def predict(self, u, i):
        if self.rg.contains_user(u) and self.rg.contains_item(i):
            u = self.rg.train_user[u]
            i = self.rg.train_item[i]
            return self.P[u].dot(self.Q[i]) + self.rg.globalMean + self.Bi[i] + self.Bu[u]
        else:
            return self.rg.globalMean

if __name__ == '__main__':
    biasmf = biasSVD()
    biasmf.train_model()
    biasmf.show_rmse()