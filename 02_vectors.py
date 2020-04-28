# 特征抽取
# 导入包
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def dict_vec():
    """
    字典数据抽取
    :return:
    """
    # 实例化
    # dict = DictVectorizer() # 返回数据类型为sparse矩阵
    dict = DictVectorizer(sparse=False) # 不以sparse矩阵返回

    #调用fit_tansform
    data = dict.fit_transform([{'city': '北京', 'temperature': 100},
                        {'city': '上海', 'temperature': 60},
                        {'city': '深圳', 'temperature': 30}])

    # print(data) # sparse矩阵，矩阵

    print(dict.get_feature_names())

    # print(data.toarray()) # one-hot编码，把数据特征转换为one-hot编码有利于对数据进行分析
    print(data) # one-hot编码，把数据特征转换为one-hot编码有利于对数据进行分析

def count_vec():
    """
    对文本进行特征值抽取
    :return:
    """

    # 实例化
    cv = CountVectorizer()

    # 提取特征值
    data = cv.fit_transform(["life is short,i like python", "life is too long,i dislike python"])

    # 特征值名称(统计所有文章当中所有的词，重复的只看做一次)
    print(cv.get_feature_names())

    # 查看数据（对每篇文章，对词的列表里面进行统计每个词出现的次数）
    print(data.toarray())

def cutwords():
    con1 = jieba.cut("10年期政府债券 -》10年期国债的收益率走势图（目前在3.5%左右）")
    con2 = jieba.cut("10年期国债收益率是每天变动的（因为国债的交易价格是每天变动的），但波动幅度一般很小，从长期看，收益率一直在百分之2-4左右波动，平均下来是3.5%左右")
    con3 = jieba.cut("大盘低于14倍市盈率时，符合[安全边际标准]")

    # 转换成列表
    list1 = list(con1)
    list2 = list(con2)
    list3 = list(con3)

    # 把列表转换成字符串
    c1 = ' '.join(list1)
    c2 = ' '.join(list2)
    c3 = ' '.join(list3)

    return c1, c2, c3

def chinese_vec():
    """
    中文特征值化
    :return:
    """

    c1, c2, c3 = cutwords()
    print(c1, c2, c3)

    # 实例化
    cv = CountVectorizer()

    # 提取特征值
    data = cv.fit_transform([c1, c2, c3])

    # 特征值名称(统计所有文章当中所有的词，重复的只看做一次)
    print(cv.get_feature_names())

    # 查看数据（对每篇文章，对词的列表里面进行统计每个词出现的次数）
    print(data.toarray())

def tfidf_vec():
    """
    单词重要程度特征值化
    :return:
    """

    c1, c2, c3 = cutwords()

    # 实例化
    tv = TfidfVectorizer()

    # 提取特征值
    data = tv.fit_transform([c1, c2, c3])

    print(tv.get_feature_names())
    print(data.toarray())

if __name__ == '__main__':
    # dict_vec()
    # count_vec()
    # chinese_vec()
    tfidf_vec()