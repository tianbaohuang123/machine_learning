from sklearn.datasets import load_iris,load_digits,load_boston,load_diabetes
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

def show_iris():
    """
    加载并返回鸢尾花数据

    名称   数量
    类别：3
    特征：4
    样本数量：150
    每个类别数量：50
    :return:
    """
    li = load_iris()
    print("获取特征值")
    print(li.data)
    print('目标值')
    print(li.target)
    print(li.DESCR)


def show_digits():
    """
    加载并返回数字数据集

    名称   数量
    类别：10
    特征：64
    样本数量：1797
    每个类别数量：50

    :return:
    """
    ld = load_digits()
    print("获取特征值")
    print(ld.data)
    print('目标值')
    print(ld.target)
    print(ld.DESCR)


def dataset_split():
    """
    数据集划分
    :return:
    """
    li = load_iris()
    # 注意返回值，训练集 train  x_train, y_train    测试机  test x_test,y_test
    x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)

    print("训练集特征值和目标值：", x_train, y_train)
    print("测试集特征值和目标值：", x_test, y_test)


def fetch_newsgroups():
    """
    分类数据集（新闻）

    下载数据集失败解决方法：
    1.
    手动下载压缩包："http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    放置以下路径："C:/Users/(用户名)/scikit_learn_data"
    注：20个新闻组数据集包含20个主题的18000个新闻组帖子
    2.修改fetch_20newsgroups中的_download_20newsgroups接口
    将以下两行注释并对archive_path显示赋值
    # logger.info("Downloading dataset from %s (14 MB)", ARCHIVE.url)
    # archive_path = _fetch_remote(ARCHIVE, dirname=target_dir)

    archive_path = r"C:/Users/tob/scikit_learn_data/20news-bydate.tar.gz"

    :return:
    """
    news = fetch_20newsgroups(subset='all')
    print(news.data)
    print(news.target)


def show_boston():
    """
    加载并返回波士顿房价书籍

    名称   数量
    目标类别：5-50
    特征：13
    样本数量：506

    :return:
    """

    lb = load_boston()

    print("获取特征值")
    print(lb.data)
    print('目标值')
    print(lb.target)
    print(lb.DESCR)


def show_diabetes():
    """
    加载并返回糖尿病数据集

    名称   数量
    目标范围：25-346
    特征：10
    样本数量：442

    :return:
    """
    ld = load_diabetes()

    print("获取特征值")
    print(ld.data)
    print('目标值')
    print(ld.target)
    print(ld.DESCR)

def knncls():
    """
    K-近邻预测用户签到位置
    特征值：x,y坐标，定位准确度，时间戳
    目标值：入住的位置id
    :return: None
    """

    # 读取数据
    # row_id：签到事件id
    # x,y: 位置坐标
    # accuracy: 定位精度
    # time: 时间戳
    # place_id：预测目标id
    data = pd.read_csv("./data/FBlocation/train.csv")
    print(data.head(10))

    # 处理数据
    # 1.缩小数据范围（节省时间）,通过查询数据筛选
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 2.处理时间戳数据，当作新的特征
    time_value = pd.to_datetime(data['time'], unit = 's')
    print(time_value)
    # 把日期格式转换为字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构造特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 删除时间戳特征
    data = data.drop(['time'], axis=1)
    print(data)

    # 3.把签到数量少于n个签到人数的位置删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据当中的特征值和目标值
    y = data['place_id']

    x = data.drop(['place_id'], axis=1)

    # 进行数据的分割：训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()

    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier(n_neighbors=5)

    # fit, pridict, score
    knn.fit(x_train, y_train)

    # 得出预测结果
    y_predict = knn.predict(x_test)

    print("预测的目标签到位置为：y_predict", y_predict)

    # 得出准确率
    print("预测的准确率：", knn.score(x_test, y_test))

    return None


def naviebayes():
    """
    朴素贝叶斯进行文本分类
    :return:
    """

    # 1.加载20类新闻数据，并进行分割
    news = fetch_20newsgroups(subset='all')

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 2.生成文章特征词
    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    # print(tf.get_feature_names())
    x_test = tf.transform(x_test)

    # 3.朴素贝叶斯estimator流程进行预估
    mlt = MultinomialNB(alpha=1.0)

    # print(x_train)

    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)

    print("预测的目标签到位置为：y_predict", y_predict)

    print("准确率为：", mlt.score(x_test, y_test))

if __name__ == '__main__':
    # 数据集API
    # show_iris()
    # show_digits()

    # 数据集划分
    # dataset_split()

    # 分类数据集
    # fetch_newsgroups()

    # 回归数据集
    # show_boston()
    # show_diabetes()

    # K-近邻算法
    # knncls()

    # 朴素贝叶斯算法
    naviebayes()