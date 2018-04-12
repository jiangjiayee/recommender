from recommender.model.mf import MF

class FunkSVD(MF):
    """
    implement FunkSVD
    """
    def __init__(self):
        super(FunkSVD, self).__init__()
        self.init_model()

    def train_model(self):
        for iteration in range(self.config.maxIteration):
            # print(f'第{i}次训练')
            self.loss = 0
            for index, line in enumerate(self.rg.train_set()):
                user, item, rating = line
                u = self.rg.train_user[user]
                i = self.rg.train_item[item]
                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                # update
                self.P[u] += self.config.lr * error * q
                self.Q[i] += self.config.lr * error * p

            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    bmf = FunkSVD()
    bmf.train_model()
    bmf.show_rmse()
