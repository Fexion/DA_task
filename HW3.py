import pandas as pd
import ezodf
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def makeClusters(df, n_clusters, colors):
    df1 = np.array(df[['CPU', 'GPU']])
    kmeans = KMeans(n_clusters=n_clusters).fit(df1)
    labels = kmeans.labels_
    values = kmeans.cluster_centers_
    #plt.figure(figsize=(13, 6))

    Clusters = [[] for i in range(n_clusters)]

    for i in range(n_clusters):
        idx = np.concatenate(np.where(labels == i))
        x = df1[idx, 0]
        y = df1[idx, 1]
        Clusters[i].append(x)
        Clusters[i].append(y)

    #for i in range(n_clusters):
    #    Clusters[i][0], Clusters[i][1] = np.array(Clusters[i][0]), np.array(Clusters[i][1])

    return np.array(Clusters), values


def performBootstrap(Clusters, values, n_iterations, feature_idx, word):
    bootstrapMeans1 = []
    bootstrapMeans2 = []
    feature1 = Clusters[0, feature_idx]
    feature2 = Clusters[1, feature_idx]

    title1, title2 = '', ''
    if values[0, 1] > 10:
        title1 = 'Cluster0'
        title2 = 'Cluster1'
    else:
        title1 = 'Cluster1'
        title2 = 'Cluster 0'

    for i in range(n_iterations):
        cur1 = np.random.choice(feature1, feature1.shape, True)
        cur2 = np.random.choice(feature2, feature2.shape, True)
        bootstrapMeans1.append(cur1.mean())
        bootstrapMeans2.append(cur2.mean())

    plt.subplot(121)
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.title(title1 + word)
    plt.hist(bootstrapMeans1)

    plt.subplot(122)
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.title(title2 + word)
    plt.hist(bootstrapMeans2)

    plt.show()
    return


def main():
    df = pd.read_excel("gradedTable.xls")
    df['Price(Ru)'] //= 100
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


    Clusters, values = makeClusters(df, 3, colors)
    idx = np.where(values[:, 0] < 5)
    Clusters = np.delete(Clusters, idx, 0)
    values = np.delete(values, idx, 0)

    iters = 1000
    performBootstrap(Clusters, values, iters, 0, ' CPU')
    performBootstrap(Clusters, values, iters, 1, ' GPU')
    return

main()