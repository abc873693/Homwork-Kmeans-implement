"""
get (data, label) from a specific dataset

label:
    Iris:
        0: setosa
        1: versicolor
        2: virginica
    Abalone:
        0: M
        1: F
        2: I
"""
import numpy as np 

class Data(object):

    def __init__(self):
        self.name = None
        raise NotImplementedError('abstract class')

    def load_data(self):
        data = np.load('./utils/data/{}.npy'.format(self.name))
        return data[:, 1:], data[:, :1].reshape(-1, )


class Iris(Data):

    def __init__(self):
        self.name = 'iris'


class Abalone(Data):

    def __init__(self):
        self.name = 'abalone'


def test():
    x, y = Iris().load_data()
    print(x.shape)
    print(y.shape)
    x, y = Abalone().load_data()
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    test()