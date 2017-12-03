import pandas as pd
import ezodf
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def read_ods(filename, sheet_no=0, header=0):
    tab = ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({col[header].value:[x.value for x in col[header+1:]]
                         for col in tab.columns()})


def performKMeans(df, n_clusters, colors):
    df1 = np.array(df[['CPU', 'GPU']])
    kmeans = KMeans(n_clusters=n_clusters).fit(df1)
    labels = kmeans.labels_
    values = kmeans.cluster_centers_
    plt.figure(figsize=(13, 6))
    grand_mean0 = df1[:, 0].mean()
    grand_mean1 = df1[:, 1].mean()
    print(grand_mean0, grand_mean1, '\n')
    for i in range(n_clusters):
        print('Cluster', str(i))
        print(values[i, 0], values[i, 1])
        print()

    for i in range(n_clusters):
        idx = np.concatenate(np.where(labels == i))
        #color = colors[i] + 'o'
        x = df1[idx, 0]
        y = df1[idx, 1]
        plt.plot(x, y, colors[i] + 'o')
        plt.plot(values[i, 0], values[i, 1],  colors[i] + 'D', markersize=15,
                 label="Cluster "+str(i))
    plt.xlabel('CPU')
    plt.ylabel('GPU')
    plt.legend()
    plt.grid()
    plt.show()
    return

def main():
    df = pd.read_excel("gradedTable.xls")
    df['Price(Ru)'] //= 100
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


    performKMeans(df, 3, colors)
    #performKMeans(df, 6, colors)

    return


main()