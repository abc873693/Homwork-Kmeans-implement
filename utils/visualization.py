import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 




def plot_scatter(data, label, predict):

    num = len(label)

    pca=PCA(n_components=2)
    data2=pca.fit_transform(data)
    pca=PCA(n_components=3)
    data3=pca.fit_transform(data)

    fig = plt.figure(figsize=(12, 8))
    colors = ['cyan', 'red', 'm', 'orange', 'blue', 'orange', 'yellow']
    ax = fig.add_subplot(223, projection = '3d')
    for i in range(150):
        ax.scatter(data3[i, 0], data3[i,1], data3[i,2], s=7, color = colors[int(label[i])])
    ax = fig.add_subplot(224, projection = '3d')
    for i in range(150):
        ax.scatter(data3[i, 0], data3[i,1], data3[i,2], s=7, color = colors[int(predict[i])])
    ax = fig.add_subplot(221)
    for i in range(150):
        plt.scatter(data2[i, 0], data2[i,1], s=7, color = colors[int(label[i])])
    ax = fig.add_subplot(222)
    for i in range(150):
        plt.scatter(data2[i, 0], data2[i,1], s=7, color = colors[int(predict[i])])
    plt.show()