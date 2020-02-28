import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_scatter3D(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for sub_data in data:
        ax.scatter(sub_data[0][:,1],sub_data[0][:,2],sub_data[1])
    plt.show()

def plot_3D_scatter_plane(data_scatter,data_planes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for sub_data in data_scatter:
        ax.scatter(sub_data[0][:,1],sub_data[0][:,2],sub_data[1])
    for sub_data in data_planes:
        ax.scatter(sub_data[0][:,0],sub_data[0][:,1],sub_data[1])
    plt.show()


if __name__ == "__main__":
    import numpy as np
    test0 = np.random.rand(1,20)
    test1 = np.random.rand(2,20)
    sub_data = list()
    sub_data.append(test1)
    sub_data.append(test0)
    data = list()
    data.append(sub_data)
    test0 = np.random.rand(1,20)
    test1 = np.random.rand(2,20)
    sub_data = list()
    sub_data.append(test1)
    sub_data.append(test0)
    data.append(sub_data)
    plot_scatter3D(data)