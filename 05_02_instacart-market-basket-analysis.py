"""
用户对物品类别的喜好细分降维

数据：
商品信息：products.csv
订单与商品信息：order_products__prior.csv
用户的订单信息：orders.csv
商品所属具体物品类别：aisles.csv

instacart：把用户分成几个类别
    用户 --- 购买的物品类别
    
            购买的物品类别
    用户1
    用户2
    用户3

步骤：
1.合并多张表到一张表当中(pd.merge)
    order_products__proir.csv: product_id, order_id
    products.csv: product_id, aisle_id
    order.csv: order_id, user_id
    aisles.csv: aisle_id, aisle
2.建立一个类似行，列数据（交叉表（特殊的分组表））
"""

from sklearn.decomposition import PCA
import pandas as pd

if __name__ == '__main__':

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
    cross = pd.crosstab(mt['user_id'],mt['aisle_id'])
    cross.head(10) # 134个特征

    # 3、进行主成分分析
    pca = PCA(n_components=0.9)
    data = pca.fit_transform(cross)
    data
    data.shape() # 27个特征