# encoding: utf-8

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, silhouette_score
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def test():
    """
    绘制一条线性回归线
    :return:
    """
    plt.figure(figsize=(10, 10))
    plt.scatter([60, 72, 75, 80, 83],  # 房子面积
                [126, 151.2, 157.5, 168, 174.3])  # 房价

    plt.show()

def linear_gs():
    """
    线性回归预测房子价格
    :return:
    """

    # 获取数据
    lb = load_boston()

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print(y_train, y_test)

    # 进行标准化处理
    # 特征值和目标值都必须进行标准化处理，实例化两个标准化PAPI
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 载入保存的模型
    # lr = joblib.load('./data/test.pkl')

    # 目标值
    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))


    # estimator预测
    # 正规方差求解方式预测结果
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_) # 权重

    # # 保存训练好的模型
    # joblib.dump(lr, './data/test.pkl')

    # 预测测试集的房子价格
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))

    print("正规方差的测试集里面每个房子的预测价格:", y_lr_predict)

    print("正规方差的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    # 梯度下降求解方式预测结果
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_) # 权重

    # 预测测试集的房子价格
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))

    print("梯度下降测试集里面每个房子的预测价格:", y_sgd_predict)

    print("梯度下降的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    # 岭回归求解方式预测结果
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print(rd.coef_)  # 权重

    # 预测测试集的房子价格
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))

    print("岭回归测试集里面每个房子的预测价格:", y_rd_predict)

    print("岭回归的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

def logistic():
    """
    逻辑回归做二分类进行癌症预测（根据细胞的属性特征）
    :return:
    """
    # 构造列标签名字
    column = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

    # 读取数据
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column)
    # print(data)

    # 缺失值处理
    data = data.replace(to_replace='?', value=np.nan)

    data = data.dropna()

    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 进行标准化处理
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)

    print(lg.coef_)

    y_predict = lg.predict(x_test)
    print('准确率：', lg.score(x_test, y_test ))

    print('精确率和召回率：', classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))

    return None

def k_means():
    # 1、合并表
    # 读取四张表的数据
    prior = pd.read_csv("./data/instacart/order_products__prior.csv")
    products = pd.read_csv("./data/instacart/products.csv")
    orders = pd.read_csv("./data/instacart/orders.csv")
    aisles = pd.read_csv("./data/instacart/aisles.csv")

    # 合并四张表到一张表 （用户-物品类别）
    _mg = pd.merge(prior, products, on=['product_id', 'product_id'])
    _mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
    mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])
    mt.head(10)

    # 2、建立行，列数据（交叉表）
    # 交叉表（特殊的分组工具） 用户 物品类别
    cross = pd.crosstab(mt['user_id'], mt['aisle_id'])
    cross.head(10)  # 134个特征

    # 3、进行主成分分析
    pca = PCA(n_components=0.9)
    data = pca.fit_transform(cross)
    data.shape()  # 27个特征

    # 减少样本数量
    x = data[:500]
    print(x.shape)

    # 假设用户一共分为四个类别
    km = KMeans(n_clusters=4)
    km.fit(x)

    predict = km.predict(x)

    # 显示聚类的结果
    plt.figure(figsize={10,10})

    # 建立四个颜色的列表
    colored = ['orange', 'green', 'blue', 'purple']
    colr = [colored[i] for i in predict]
    plt.scatter(x[:, 1], x[:, 20], color=colr)
    plt.xlabel("1")
    plt.ylabel("20")
    plt.show()

    # 评判聚类效果：轮廓系数
    silhouette_score(x, predict)

if __name__ == '__main__':
    # linear_gs()
    logistic()