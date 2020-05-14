# coding: utf-8
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一张图（包含了一组op和tensor），上下文环境
# op：只要使用tensorflow的API定义的函数都是OP
# tensor：就是指代的数据
g = tf.Graph()

print(g)
with g.as_default():
    c = tf.constant(11.0)
    print(c.graph)

# 实现一个加法

a = tf.constant(5.0)
b = tf.constant(6.0)

print(a, b)

sum1 = tf.add(a, b)

# 有重载的机制，默认会给运算符重载成op类型
var1 = 2.0
sum2 = var1 + a
print(sum2)

# 图
graph = tf.get_default_graph()

print(graph)

print(sum1)

# 训练模型
# 实时的提供数据去进行训练
plt = tf.placeholder(tf.float32, [None, 3])
print(plt)

# 会话只能运行一张图，可以在会话当中指定图去运行
# 只要有会话的上下文环境，就可以使用eval()，查看变量值
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sess.run(plt, feed_dict={plt: [[1, 2, 3],[4, 5, 6]]})))
    print(sess.run(sum1))
    print(sess.run(sum2))
    print(sum1.eval())
    print(a.graph)
    print(sum1.graph)