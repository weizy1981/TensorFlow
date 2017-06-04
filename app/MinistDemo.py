import app.input_data as input_data
import tensorflow as tf

# 导入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 创建占位符
x = tf.placeholder("float", [None, 784])

'''一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。
它们可以用于计算输入值，也可以在计算中被修改。
对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。'''
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 创建模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型
# 为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：
y_ = tf.placeholder("float", [None, 10])
# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化我们创建的变量
init = tf.global_variables_initializer()
# 启动我们的模型，并且初始化变量
sess = tf.Session()
sess.run(init)
# 训练模型，这里我们让模型循环训练1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 训练模型

#评估我们的模型
#用 tf.equal 来检测我们的预测是否真实标签匹配
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#把布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#评估我们的模型
