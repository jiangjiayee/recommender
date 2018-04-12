import numpy as np

from recommender.configx.configx import Configx
from collections import defaultdict

class RatingGetter(object):

    """
    read ratings data
    """

    def __init__(self):
        super(RatingGetter, self).__init__()
        self.config = Configx()

        self.train_user = {}
        self.train_item = {}
        self.all_User = {}
        self.all_Item = {}
        self.id2user = {}
        self.id2item = {}
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)  # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = defaultdict(dict)  # used to store the test set by hierarchy item:[user,rating]
        self.train_data = []
        self.test_data = []
        self.trainSetLength = 0
        self.testSetLength = 0
        self.userMeans = {}
        self.itemMeans = {}
        self.globalMean = 0.0


        self.split_train_test()
        self.gen_data()
        self.get_statistics()
        self.train_set()
        self.test_set()

    def split_train_test(self):
        np.random.seed(self.config.random_state)
        # print(self.config.path)
        with open(self.config.path) as f:
            f.readline()
            for index, line in enumerate(f):
                rand_num = np.random.rand()
                if rand_num < self.config.size:
                    self.train_data.append(line)
                else:
                    self.test_data.append(line)
        # print(self.train_data)

    def train_set(self):
        for line in self.train_data:
            u, i, r, t = line.split(',')
            yield (u, i, float(r))

    def test_set(self):
        for line in self.test_data:
            u, i, r, t = line.split(',')
            yield (u, i, float(r))

    def gen_data(self):
        for index, line in enumerate(self.train_set()):
            u, i, r = line

            if u not in self.train_user:
                self.train_user[u] = len(self.train_user)
                self.id2user[self.train_user[u]] = u
            if i not in self.train_item:
                self.train_item[i] = len(self.train_item)
                self.id2item[self.train_item[i]] = i

            self.trainSet_u[u][i] = r
            self.trainSet_i[i][u] = r
            self.trainSetLength = index + 1
        self.all_User.update(self.train_user)
        self.all_Item.update(self.train_item)

        for index, line in enumerate(self.test_set()):
            u, i, r = line
            if u not in self.train_user:
                self.all_User[u] = len(self.all_User)
            if i not in self.train_item:
                self.all_Item[i] = len(self.all_Item)
            self.testSet_u[u][i] = r
            self.testSet_i[i][u] = r
            self.testSetLength = index + 1

    def get_statistics(self):
        global_rating = 0.0
        global_length = 0.0
        for u in self.train_user:
            u_total = sum(self.trainSet_u[u].values())
            u_length = len(self.trainSet_u[u])
            global_rating += u_total
            global_length += u_length
            self.userMeans[u] = u_total / u_length

        for i in self.train_item:
            self.itemMeans = sum(self.trainSet_i[i].values()) / float(len(self.trainSet_i[i]))

        if global_length == 0:
            self.globalMean = 0
        else:
            self.globalMean = global_rating / global_length

    def contains_user(self, u):
        if u in self.train_user:
            return True
        else:
            return False

    def contains_item(self, i):
        if i in self.train_item:
            return True
        else:
            return False

    def contains_user_item(self, user, item):
        if user in self.trainSet_u:
            if item in self.trainSet_u[user]:
                return True
        return False

    def get_train_size(self):
        return (len(self.train_user), len(self.train_item))


# if __name__ == '__main__':
#     rg = RatingGetter()
#     print(rg.get_train_size())

