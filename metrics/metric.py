import math

class Matric(object):

    def __init__(self):
        super(Matric, self).__init__()

    @staticmethod
    def mae(res):
        error = 0
        count = 0
        for i in res:
            error += abs(i[0]-i[1])
            count += 1
        if count == 0:
            return error
        return float(error) / count

    @staticmethod
    def rmse(res):
        error = 0
        count = 0
        for i in res:
            error += abs(i[0] - i[1]) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(float(error) / count)