import pandas
import random
import math
import operator

class UserCF:

    def __init__(self):
        self.train_table = {}
        self.item_users = {}
        self.train = []
        self.test = []
        self.test_table = {}
        self.W = {}

    def loadDataSet(self, path):
        test = []
        train = []
        with open(path,'r') as f:
            f.readline()
            for line in f:
                user, item, ratings, timestamp = line.split(',')
                if random.random() > 0.8:
                    train.append([user, item])
                else:
                    test.append([user, item])
            f.close()
        self.train = train
        self.test = test


    def UserSimilarity(self):
        # build table for train_table
        train_table = dict()
        for user, item in self.train:
            if user not in train_table.keys():
                train_table[user] = set()
            train_table[user].add(item)
        self.train_table = train_table

        test_table = dict()
        for user, item in self.test:
            if user not in test_table.keys():
                test_table[user] = set()
            test_table[user].add(item)
        self.test_table = test_table

        # build inverse table for item_users
        item_users = dict()
        for user, items in train_table.items():
            for item in items:
                if item not in item_users.keys():
                    item_users[item] = set()
                item_users[item].add(user)
        self.item_users = item_users

        # calculate co-rated items
        C = dict()
        N = dict()
        for item, users in item_users.items():
            for u in users:
                if u not in N:
                    N[u] = 0
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    if u not in C:
                        C[u] = dict()
                    if v not in C[u]:
                        C[u][v] = 0
                    C[u][v] += 1

        # calculate final similarity matrix W
        W = dict()
        for u, related_users in C.items():
            if u not in W:
                W[u] = dict()
            for v,number in related_users.items():
                if v not in W[u]:
                    W[u][v] = 0.0
                W[u][v] = number / math.sqrt(N[u] * N[v])
        self.W = W

    def Recommend(self, user, K):
        rank = dict()
        interacted_items = self.train_table[user]
        # print(W[user])
        for v, wuv in sorted(self.W[user].items(), key=operator.itemgetter(1), reverse=True)[0:K]:
            for i in self.train_table[v]:
                if i in interacted_items:
                    continue
                if i not in rank.keys():
                    rank[i] = 0.0
                # print(wuv, '|', i)
                rank[i] += float(wuv) * 1.0
        return rank

    def Precision(self, N):
        hit = 0
        all = 0
        for user in self.train_table.keys():
            tu = self.test_table[user]
            rank = self.Recommend(user, N)
            for item, pui in sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:10]:
                if item in tu:
                    hit += 1
            all += N
        return hit / (all * 1.0)



usercf = UserCF()
usercf.loadDataSet('../source/ml-latest-small/ratings.csv')
usercf.UserSimilarity()
precision = usercf.Precision(10)
print(precision)

