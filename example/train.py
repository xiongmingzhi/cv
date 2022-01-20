import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # sklearn中已经废弃cross_validation,将其中的内容整合到model_selection中
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

pb_file_path = os.getcwd()
# a=[0,1,2,3,4,5,6,7,8]
# print(a[2:-1])  #始下标（默认从0开始）、指定结束下标（切片不会含该元素）

datafile = './data.xlsx'  # 改成你的路径
pd_data = pd.read_excel(datafile)
data = np.array(pd_data)
# print(data)
label = data[:, -1]  # 表格最后一列为标签
print(label.shape)  # (193，)
label = label.reshape(-1, 1)  # from (193，) to  (193,1)
train_data = data[:, 2:-1]  # 第0列是序号，第1列是图片名称，第2到-1（最后一列 ,不含结束列）才是图片的特征

X_train, X_test, label_train, label_test = train_test_split(train_data, label, train_size=0.8)  # 将数据集分为训练集和验证集
print("data.shape:", data.shape, "  Xtrain.shape:", X_train.shape, "  Xtest.shape:", X_test.shape)
print("label.shape:", label.shape, "  Ytrain.shape:", label_train.shape, "  label_test.shape:", label_test.shape)
# data.shape: (193, 9)   Xtrain.shape: (154, 6)   Xtest.shape: (39, 6)
# label.shape: (193, 1)   Ytrain.shape: (154, 1)   label_test.shape: (39, 1)

# 定义BP神经网络的输入输出
INPUT_NODE = 6
OUTPUT_NODE = 1


def test_BP(hide_node=10):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE])
    y = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
    # 数据预处理，归一化
    # X_train=preprocessing.scale(X_train)
    # X_test=preprocessing.scale(X_test)
    # layer1
    predict_1 = tf.layers.dense(inputs=x, units=hide_node, activation=tf.nn.relu, name="layer1")
    # layer2
    predict_2 = tf.layers.dense(inputs=predict_1, units=OUTPUT_NODE, activation=tf.nn.sigmoid, name="layer2")
    # 代价函数
    loss = tf.reduce_mean(tf.square(y - predict_2), name='loss')
    saver = tf.train.Saver()

    # 定义优化器。使用动量法 也可以使用随机梯度下降法等
    # train_step = tf.train.MomentumOptimizer(0.05,0.05).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("init_results:", sess.run(loss, feed_dict={x: X_train, y: label_train}))
        for i in range(10001):
            sess.run(train_step, feed_dict={x: X_train, y: label_train})
            # print("end_results:", sess.run(loss, feed_dict={x: X_train, y: label_train}))
            if i % 2000 == 0:
                train_results = sess.run(loss, feed_dict={x: X_train, y: label_train})
                val_results = sess.run(loss, feed_dict={x: X_test, y: label_test})
                print("hide_node:", hide_node, "_train_accuraty_", i, ":", 1 - train_results, end=" ")
                print("_val_accuraty_", i, ":", 1 - val_results)

        # 保存pb pbtxt
        import os
        pb_file_path = os.getcwd()
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                   output_node_names=['layer2/Sigmoid'])  # 输出的节点名称
        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['layer2/BiasAdd'])
        # 写入序列化的 PB 文件
        tf.train.write_graph(sess.graph.as_graph_def(), pb_file_path, 'tf_BP_regress.pbtxt', as_text=True)
        # tf.train.write_graph(sess.graph_def, pb_file_path, 'model.pbtxt', as_text=True)
        with tf.gfile.FastGFile('tf_BP_regress.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # 将预测结果写入到excel
        import xlsxwriter  as xlwt
        workbook = xlwt.Workbook(pb_file_path + '/predict.xls')
        worksheet = workbook.add_worksheet()
        pre = sess.run(predict_2, feed_dict={x: X_test, y: label_test})
        results = pre - label_test
        worksheet.write_column('A1', pre)
        worksheet.write_column('B1', label_test)
        worksheet.write_column('C1', results)
        pre = sess.run(predict_2, feed_dict={x: X_train, y: label_train})
        results = pre - label_train
        worksheet.write_column('D1', pre)
        worksheet.write_column('E1', label_train)
        worksheet.write_column('F1', results)
        workbook.close()


test_BP(5)