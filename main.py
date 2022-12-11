import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def display(x, y, title, ylabel='None'):
    plt.figure(figsize=(16, 8))
    plt.plot(x, y)
    plt.xlabel('K')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def silhouette(K, silhouette):
    K.append(len(silhouette))
    ylabel = 'silhouette score'
    title = 'silhouette'
    display(K, silhouette, title, ylabel)


# k-means 是 unsupervised

def main():
    df = pd.read_csv('four_D_points.csv')
    # print(df)
    attributes = ['1', '2', '3', '4']

    df.columns = [i for i in attributes]  # 添加column的名稱

    X = df.loc[:, :].values  # 取得所有欄位的值
    X = StandardScaler().fit_transform(X)  # 標準化

    pca = PCA(n_components=2)  # 使用PCA降維(維度: 2)
    pca = pca.fit_transform(X)

    K = []
    silhouette = []
    for k in range(2, 11):

        k_means = KMeans(n_clusters=k, n_init='auto')
        k_means.fit(pca)
        k_means_y = k_means.predict(pca)

        plt.figure(figsize=(16, 8))
        plt.title(f"k-means: {k} groups")
        # 降到2D之後，就可以把第一個主成分當成X、第2個主成分當成Y畫圖
        plt.scatter(pca[:, 0], pca[:, 1], c=k_means_y)

        centroid = k_means.cluster_centers_  # 找出K個聚類的中心
        plt.scatter(centroid[:, 0], centroid[:, 1], c='red')
        labels = k_means.labels_

        silhouette.append(silhouette_score(pca, labels)
                          )  # 計算輪廓係數，可以發現K=3，為最高分

        plt.show()
        silhouette(K, silhouette)


main()
