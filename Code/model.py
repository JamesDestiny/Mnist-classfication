import os
import tensorflow as tf

from tensorflow.keras.utils import plot_model

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
model = tf.keras.models.load_model("../Predict/model")
    #生成一个模型图，第一个参数为模型，第二个参数为要生成图片的路径及文件名，还可以指定两个参数：
    #show_shapes:指定是否显示输出数据的形状，默认为False
    #show_layer_names:指定是否显示层名称，默认为True
plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=False)