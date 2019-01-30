# coding=utf-8
# convert keras models to *.pb files

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np

# 数据预处理
batch_size = 128
num_classes = 10
epochs = 30
img_rows, img_cols = 28, 28
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# keras模型搭建
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
 
model.summary()

# 训练模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
 
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# keras模型保存为pb文件
sess = K.get_session()
 
frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    output_node_names=["y/Softmax"])
 
# 保存图为pb文件
# with open('model.pb', 'wb') as f:
#     f.write(frozen_graph_def.SerializeToString())
 
tf.train.write_graph(frozen_graph_def, 'model', 'test_model.pb', as_text=False)

# 模型预测
a = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.517647, 0.839216, 0.992157, 0.996078, 0.992157, 0.796079, 0.635294, 0.160784,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.556863, 0.796079, 0.796079, 0.992157, 0.988235, 0.992157,
     0.988235, 0.592157, 0.27451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.996078, 0.992157, 0.956863,
     0.796079, 0.556863, 0.4, 0.321569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.67451,
     0.988235, 0.796079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0823529, 0.87451,
     0.917647, 0.117647, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.478431, 0.992157,
     0.196078, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.482353, 0.996078, 0.356863,
     0.2, 0.2, 0.2, 0.0392157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0823529, 0.87451, 0.992157,
     0.988235, 0.992157, 0.988235, 0.992157, 0.67451, 0.321569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0.0823529, 0.839216, 0.992157, 0.796079, 0.635294, 0.4, 0.4, 0.796079, 0.87451, 0.996078, 0.992157, 0.2, 0.0392157,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.239216, 0.992157, 0.670588, 0, 0, 0, 0, 0, 0.0784314, 0.439216,
     0.752941, 0.992157, 0.831373, 0.160784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0.4, 0.796079, 0.917647, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0784314,
     0.835294, 0.909804, 0.321569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.243137,
     0.796079, 0.917647, 0.439216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0784314,
     0.835294, 0.988235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.992157, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.160784, 0.913726, 0.831373, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0.443137, 0.360784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.121569, 0.678431, 0.956863, 0.156863,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.321569, 0.992157, 0.592157, 0, 0, 0, 0, 0, 0, 0.0823529, 0.4, 0.4, 0.717647,
     0.913726, 0.831373, 0.317647, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.321569, 1.0, 0.992157, 0.917647, 0.596078, 0.6,
     0.756863, 0.678431, 0.992157, 0.996078, 0.992157, 0.996078, 0.835294, 0.556863, 0.0784314, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0.278431, 0.592157, 0.592157, 0.909804, 0.992157, 0.831373, 0.752941, 0.592157, 0.513726,
     0.196078, 0.196078, 0.0392157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0])
 
print(model.predict(a.reshape(1, img_rows, img_cols, 1)))
# print(model.input.name)
# print(model.output.name)

# python调用pb文件
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
 
    with open('model/test_model.pb', "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
 
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
 
        input_x = sess.graph.get_tensor_by_name("x_input:0")
        output = sess.graph.get_tensor_by_name("y/Softmax:0")
 
        print(sess.run(output, feed_dict={input_x: a.reshape(1, img_rows, img_cols, 1)})）

