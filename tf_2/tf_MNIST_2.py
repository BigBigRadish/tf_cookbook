# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.model_pruning.python.learning import train_step
from sys import argv
#mnist数据集相关的常熟
INPUT_NODE=784#输入层的节点数
OUTPUT_NODE=10#输出层的节点数，需要区分0-9这10个数字
#配置神经网络参数
LAYER1_NODE=500#只使用一个隐藏层，即500个节点
BATCH_SIZE=100#一个batch的数据个数
LEARNING_RATE_BASE=0.8#初始学习率
learning_rate_decay=0.99#学习率的衰减率
REGULARIZATION_RATE=0.0001#正则化项在损失函数中的入系数
TRAINING_STEPS=30000#训练论数
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率
'''
一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果,在这里定义了一个使用RELU激活函数的三层全连接网络，通过加入隐藏层实现多层网络结构
这样方便在测试时使用滑动平均模型
'''
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有使用滑动平均时，直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层的前向传播结果，这里使用了ReLU函数
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        #计算输出层的前向传播结果，因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入softmax，并不会影响预测结果。因为预测时使用的是不同类别对应节点输出值的相对大小，对最后分类结果没有影响，所以向前传播可以不加入softmax
        return tf.matmul(layer1,weights2)+biases2
    else:
        #首先使用avg_class.average函数计算滑动平均值#然后在计算相应的前向传播结果
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
#训练模型过程
def train(mnist):
    x=tf.placeholder(tf.float32, [None,INPUT_NODE], name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    #生成隐藏层的参数
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    #计算在当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均的类为None，所以不会使用参数的滑动平均
    y=inference(x, None, weights1, biases1, weights2, biases2)
    '''
             定义存储训练论数的变量，这个变量不需要计算滑动平均值，所以制定这个变量为不可训练的变量（trainacle=False）
    ，  训练轮数的变量都是不可训练的。
    '''
    global_step=tf.Variable(0,trainable=False)
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类，，加快训练早期变量的更新速度
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #在所有代表神经网络参数的变量上使用滑动平均，tf.trainable_variables返回的就是图上的集合
    #GraphKeys.TRAINABLE_VARIABLES中的元素，这个集合中的变量都是阔以训练的
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    average_y=inference(x, variable_averages, weights1, biases1, weights2, biases2)
    #使用tensorflow中的sparse_softmax_cross_entropy_with_logits函数计算交叉熵，可以加速交叉熵的计算，如果只有一个正确答案，需要用tf.argmax函数得到对应类别的标号
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #计算batch中所有样例的交叉熵平均值
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    #计算L2正则化系数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。一般只计算权重损失
    regularization=regularizer(weights1)+regularizer(weights2)
    #总损失等于交叉损失+正则化损失
    loss=cross_entropy_mean+regularization
    #设置指数衰减的学习率
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,learning_rate_decay)
    #使用梯度下降来优化损失函数
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #在神经网络中没过一遍数据，需要通过反向传播更新参数，又要更新一个参数的滑动平均值
    #为了一次完成多个操作，tensorflow提供两种方式tf.control_dependencies和tf.group
    #train_op=tf.group(train_step,variable_averages_op)和下面等价
    with  tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name="train")
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#将数值转为实数
#初始化会话，开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #准备验证数据，一般在训练中会通过验证数据判断停止条件与评价训练效果
        validate_feed = {x: mnist.validation.images, y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        #迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training steps,validation accuracy"
                      "using average model is %g"%(i,validate_acc))
                #产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            #训练结束后，在测试数据集上检测神经网络模型的最终正确率
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("After %d training steps,test accuracy"
              "using average model is %g"%(TRAINING_STEPS,test_acc))    
#主程序入口
def main(argv=None):
    mnist=input_data.read_data_sets('./mnist/',one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()
    
    