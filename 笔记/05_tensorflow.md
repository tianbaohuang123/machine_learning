## tensorflow

1. 深度学习介绍

   深度学习，如**深度神经网络、卷积神经网络**和递归神经网络已被应用计算机视觉、语音识别、自然语言处理、音频识别与生物信息学等领域并获取了极好的效果。

   |      | 机器学习         | 深度学习                   |
   | ---- | ---------------- | -------------------------- |
   | 算法 | 神经网络（简单） | 神经网络（深度）           |
   |      | 回归             | 图像：卷积神经网络         |
   |      |                  | 自然语言处理：循环神经网络 |

   ​

   应用：

   * **图像理解**
   * 语音识别
   * **自然语言处理**
   * 机器自主

2. 认识Tensorflow

   * Google Brain计划产物
   * 应用于AlphaGo, Gmail, Google Maps等1000多个产品
   * 于2015年11月开源，2017年2月发布1.0版本
   * 架构师 Jeff Dean
     * 领导实现MapReduce、BigTable、Spanner

   tensorflow

   * 全面的深度学习框架
   * 支持非常全面
   * 不是专门为客户端设计

   Caffe and Caffe2

   * 精简高效
   * Caffe2专门为客户端设计

   Tensorflow特点

   * 真正的可移植性
     * 引入各种计算设备的支持包括CPU/GPU/TPU，以及能够很好地运行在移动端，如安卓设备、ios、树莓派等等
   * 多语言支持
     * Tensorflow 有一个合理的c++使用界面，也有一个易用的python使用界面来构建和执行你的graphs，你可以直接写python/c++程序。
   * 高度的灵活性与效率
     * TensorFlow是一个采用数据流图（data flow graphs），用于数值计算的开源软件库，能够灵活进行组装图，执行图。随着开发的进展，Tensorflow的效率不断在提高
   * 支持
     * TensorFlow 由谷歌提供支持，谷歌投入了大量精力开发 TensorFlow，它希望 TensorFlow 成为机器学习研究人员和开发人员的通用语言

3. Tensorflow的安装

   Linux/ubuntu:

   * python2.7: pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl
   * python3.5: pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp35-cp35m-linux_x86_64.whl

   Maxos:

   * python2: pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl
   * python3: pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl

4. Tensorflow初体验

   计算密集型：cpu计算（tensorflow）

   IO密集型：django,scrapy,http请求,磁盘操作等等

5. **Tensorflow进阶**

   * 图

     * **图默认已经注册，一组表示 tf.Operation计算单位的对象和tf.Tensor表示操作之间流动的数据单元的对象** 

     * 获取调用：

       1. tf.get_default_graph()
       2. op、sess或者tensor 的graph属性

     * 图的创建

       * tf.Graph()

       * 使用新创建的图

         ```python
         g = tf.Graph()
         with g.as_default():
             a = tf.constant(1.0)
             assert c.graph is g
         ```

     * op

     | 类型           | 示例                                                 |
     | -------------- | ---------------------------------------------------- |
     | 标量运算       | Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal   |
     | 向量运算       | Concat, Slice, Splot, Constant, Rank, Shape, Shuffle |
     | 矩阵运算       | Matmul, Matrixinverse, MatrixDeterminant             |
     | 带状态的运算   | Variable, Assign, AssignAdd                          |
     | 神经网络组件   | SoftMax, Sigmoid, ReLU, Convolution2D, MaxPooling    |
     | 存储、恢复     | Save, Restore                                        |
     | 队列及同步运算 | Enqueue, Dequeue, MutexAcquire, MutexRelease         |
     | 控制流         | Merge, Switch, Enter, Leave, NextIteration           |

     * 会话

       * 1. 运行图的结构
         2. 分配资源计算
         3. 掌握资源（**变量**的资源，队列，线程）

         * 使用

           * tf.Session()

             **运行TensorFlow操作图的类，使用默认注册的图（可以指定运行图）** 


           * 会话资源

             会话可能拥有很多资源，如 tf.Variable，tf.QueueBase和tf.ReaderBase，会话结束后需要进行资源释放

             1. sess = tf.Session()     sess.run(...)     sess.close() 
             2. 使用上下文管理器

             with tf.Session() as sess: 

               	sess.run(...)

           * config=tf.ConfigProto(log_device_placement=True)

             可以看到程序在哪里进行运算

           * 交互式：tf.InteractiveSession()

             一般在命令行使用

         * 会话的run()方法

           * run(fetches,feed_dict=None,graph=None)

             运行ops和计算tensor

             * 嵌套列表，元组，

               namedtuple，dict或OrderedDict(**重载的运算符也能运行**)

             * feed_dict 允许调用者覆盖图中指定张量的值,提供给placeholder使用

           * 返回值异常

             RuntimeError：如果它Session处于无效状态（例如已关闭）。

             TypeError：如果fetches或feed_dict键是不合适的类型。

             ValueError：如果fetches或feed_dict键无效或引用Tensor不存在。

         * Feed操作

           * 意义：在程序执行的时候，不确定输入的是什么，提前**“占个坑”**
           * 语法：placeholder提供占位符， run的时候通过feed_dict指定参数

       * 张量

         * 1. 张量的阶和数据类型

              * Tensorflow基本的数据格式
              * 一个类型化的N维度数组（**tf.Tensor**）
              * 三部分：名字，形状，数据类型

              ​

              张量的阶

              | 阶   | 数学实例 | Python         | 例子                                                 |
              | ---- | -------- | -------------- | ---------------------------------------------------- |
              | 0    | 纯量     | （只有大小）   | s = 483                                              |
              | 1    | 向量     | （大小和方向） | v = [1.1, 2.2, 3.3]                                  |
              | 2    | 矩阵     | （数据表）     | m = [[1,2,3],[4,5,6],[7,8,9]]                        |
              | 3    | 3阶张量  | （数据立体）   | t = [[[2],[4],[6]],[[8],[10],[12]],[[14],[16],[18]]] |
              | n    | n阶      | （自己构想）   | ...                                                  |

              张量的数据类型

              | 数据类型     | Python类型   | 描述                                               |
              | ------------ | ------------ | -------------------------------------------------- |
              | DT_FLOAT     | tf.float32   | 32位浮点数                                         |
              | DT_DOUBLE    | tf.float64   | 64位浮点数                                         |
              | DT_INT64     | tf.int64     | 64位有符号整型                                     |
              | DT_INT32     | tf.int32     | 32位有符号整型                                     |
              | DT_INT16     | tf.int16     | 16位有符号整型                                     |
              | DT_INT8      | tf.int8      | 8位有符号整型                                      |
              | DT_UINT8     | tf.uint8     | 8位无符号整型                                      |
              | DT_STRING    | tf.string    | 可变长度的字节数组，每一个张量元素都是一个字节数组 |
              | DT_BOOL      | tf.bool      | 布尔型                                             |
              | DT_COMPLEX64 | tf.complex64 | 由两个32位浮点数组成的复数：实数和虚数             |
              | DT_QINT32    | tf.qint32    | 用于量化Ops的32位有符号整型                        |
              | DT_QINT8     | tf.qint8     | 用于量化Ops的8位有符号整型                         |
              | DT_QUINT8    | tf.quint8    | 用于量化Ops的8位无符号整型                         |

              张量属性

              * graph：张量所属的默认图
              * op：张量的操作名
              * name：张量的字符串描述
              * shape：张量形状

              张量的动态形状与静态形状

              * TensorFlow中，张量具有静态形状和动态形状

              * 静态形状

                创建一个张量或者由操作推导出一个张量时,初始状态的形状

                * tf.Tensor.get_shape:获取静态形状
                * tf.Tensor.set_shape():更新Tensor对象的静态形状，通常用于在不能直接推断的情况下

              * 动态形状：

                一种描述原始张量在执行过程中的一种形状

                * tf.reshape:创建一个具有不同动态形状的新张量

              * 要点

                1. 转换静态形状的时候，1-D到1-D，2-D到2-D，不能跨阶数改变形状
                2. 对于已经固定或者设置静态形状的张量／变量，不能再次设置静态形状
                3. tf.reshape()动态创建新张量时，元素个数不能不匹配

           2. 张量操作

           关闭警告

           ```python
           import os
           os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
           ```

           ​

6. 案例：实现线性回归