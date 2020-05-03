from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

"""
特征值处理：
1.归一化：MinMaxScaler
2.标准化：StandardScaler
3.缺失值处理：SimpleImputer
"""

def min_max_scaler():
    """
    归一化
    :return:
    """

    # 实例化
    # mm_scaler = MinMaxScaler() # 默认归一范围（0，1）
    mm_scaler = MinMaxScaler(feature_range=(2,3))
    data = mm_scaler.fit_transform([[90,2,10,40],
                                    [60,4,15,45],
                                    [75,3,13,46]])
    print(data)

def std_scaler():
    """
    标准化处理
    :return:
    """

    # 实例化
    std = StandardScaler()
    data = std.fit_transform([[1.,-1.,3.],
                             [2.,4.,2.],
                             [4.,6.,-1.]])
    print(data)

def ip_imputer():
    """
    缺失值处理
    :return:
    """

    # 实例化
    ip = SimpleImputer(missing_values=np.nan, strategy='mean')

    # 缺失值处理
    data = ip.fit_transform([[1, 2], [np.nan, 3], [7,6]])
    print(data)

if __name__ == '__main__':
    # min_max_scaler()
    # std_scaler()
    ip_imputer()