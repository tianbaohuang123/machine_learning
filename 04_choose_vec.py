from sklearn.feature_selection import  VarianceThreshold

"""
特征选择方法：
1.过滤式：VarianceThreshold
2.嵌入式：
3.包裹式：
"""

def var():
    """
    特征选择-删除低方差的特征
    :return:
    """

    # 实例化
    var = VarianceThreshold(threshold=0.0) # 根据实际选取方差下限，这里删除相等的特征(整个特征值都相同，无有效信息)

    data = var.fit_transform([[0, 2, 0, 3],
                              [0, 1, 4, 3],
                              [0, 1, 1, 3]])
    print(data)
    return None

if __name__ == '__main__':
    var()