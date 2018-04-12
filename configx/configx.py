class Configx(object):

    """
    configurate the global parameters and hyper parameters
    """

    def __init__(self):
        super(Configx,self).__init__()

        self.path = '../source/ml-latest-small/ratings.csv'
        self.random_state = 0
        self.size = 0.8
        self.min_val = 0.5  # 0.5 1.0
        self.max_val = 5.0  # 4.0 5.0

        # Hyper parameters
        self.factor = 5     # 隐因子个数
        self.threshold = 1e-4   # 阈值
        self.rate = 0.01  # 学习率
        self.lambdaP = 0.001  # 0.02
        self.lambdaQ = 0.001  # 0.02
        self.maxIteration = 100
        self.lr = 0.01