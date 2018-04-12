import numpy as np
import matplotlib.pyplot as plt

from recommender.reader.ratings import RatingGetter
from recommender.configx.configx import Configx
from recommender.metrics.metric import Matric

class MF(object):
    """
    base class for matrix factorization
    """

    def __init__(self):
        super(MF, self).__init__()
        self.rg = RatingGetter()
        self.config = Configx()
        self.iter_rmse = []
        self.iter_mae = []

    def init_model(self):
        self.P = np.random.rand(self.rg.get_train_size()[0], self.config.factor)
        self.Q = np.random.rand(self.rg.get_train_size()[1], self.config.factor)
        self.loss, self.lastLoss = 0.0, 0.0

    def train_model(self):
        pass

    def predict(self, u, i):
        if self.rg.contains_user(u) and self.rg.contains_item(i):
            return self.P[self.rg.train_user[u]].dot(self.Q[self.rg.train_item[i]])
        elif self.rg.contains_user(u) and not self.rg.contains_item(i):
            return self.rg.userMeans[u]
        elif not self.rg.contains_user(u) and self.rg.contains_item(i):
            return self.rg.itemMeans[i]
        else:
            return self.rg.globalMean

    def predict_model(self):
        result = []
        for index, line in enumerate(self.rg.test_set()):
            user, item, rating = line
            prediction = self.predict(user, item)
            prediction = self.checkRatingBoundary(prediction)
            result.append([rating, prediction])
        rmse = Matric.rmse(result)
        mae = Matric.mae(result)
        self.iter_rmse.append(rmse)
        self.iter_mae.append(mae)
        return rmse, mae
        # print(f'最终误差RMSE为{rmse}')

    def checkRatingBoundary(self, prediction):
        if prediction > self.config.max_val:
            return self.config.max_val
        elif prediction < self.config.min_val:
            return self.config.min_val
        else:
            return round(prediction, 3)

    def isConverged(self, iteration):
        deltaLoss = self.lastLoss - self.loss
        rmse, mae = self.predict_model()
        print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f rmse=%.5f mae=%.5f' % \
              (self.__class__, iteration, self.loss, deltaLoss, self.config.lr, rmse, mae))
        # check if converged
        converged = abs(deltaLoss) < self.config.threshold
        self.lastLoss = self.loss
        return converged

    def show_rmse(self):
        '''
        show figure for rmse and epoch
        '''
        nums = range(len(self.iter_rmse))
        plt.plot(nums, self.iter_rmse, label='RMSE')
        plt.plot(nums, self.iter_mae, label='MAE')
        plt.xlabel('# of epoch')
        plt.ylabel('metric')
        plt.title(self.__class__)
        plt.legend()
        plt.show()
