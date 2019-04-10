from utils.datasets import Iris, Abalone
from utils.visualization import plot_scatter
import numpy as np
from copy import deepcopy
import numpy as np 
"""
要改距離計算方式，更改有 np.linalg.norm 的地方
載入資料集：
    data, labels = Abalone().load_data() 
    data, labels = Iris().load_data() 
"""

class Kmeans(object):

    def __init__(self, data, k):
        self.n_clusters = 3
        self.data = data
        self.label = None
        self.__clusters = np.zeros(self.data.shape[0])
        self.__distances = np.zeros((self.data.shape[0],k))

        self.centers = self.__init_center()
        self.__old_centers = np.zeros(self.centers.shape) # to store old centers
        self.__new_centers = deepcopy(self.centers) 



    def __init_center(self):
        return (np.random.randn(self.n_clusters, self.data.shape[1]) *
                                np.std(self.data.astype(int), axis = 0) +
                                np.mean(self.data, axis = 0))


    def __assign(self):
        for i in range(self.n_clusters):
            self.__distances[:,i] = np.linalg.norm(self.data.astype(int) - self.__new_centers[i].astype(int), axis=1)
        self.__clusters = np.argmin(self.__distances, axis = 1)

    
    def __update(self):
        self.__old_centers = deepcopy(self.__new_centers)
        for i in range(self.n_clusters):
            self.__new_centers[i] = np.mean(self.data[self.__clusters == i], axis=0)


    # 執行
    def fit(self):
        centers_change = np.linalg.norm(self.__new_centers - self.__old_centers)

        #while centers_change != 0:
        for iteration in range(20):
            print(np.sum(np.mean(self.__distances, axis = 1)))
            self.__assign()
            self.__update()
            centers_change = np.linalg.norm(self.__new_centers - self.__old_centers)

        self.centers = self.__new_centers
        self.label = self.__clusters


    # 計算精準度，輸入真實 label
    def accuracy(self, lab):
        correct = 0
        for i in range(lab.max() + 1):
            correct += np.bincount(self.label[lab == i]).max()
        return correct / len(lab)

    # 畫圖，輸入真實 label
    def scatter(self, lab):
        plot_scatter(self.data, self.label, lab)




if __name__ == "__main__":

    data, labels = Abalone().load_data()
    result = Kmeans(data, 3)
    result.fit()
    print(result.centers)
    print(result.label)  # 執行結果的 label
    print(result.accuracy(labels)) 
    result.scatter(labels)


