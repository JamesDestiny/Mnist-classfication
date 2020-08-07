from CNN.cnn import CNN
from dataset.mnist import MNISTLoader
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def pre(data=None):
    num_epochs = 0.1
    batch_size = 50
    learning_rate = 0.001
    model = CNN()#模型创建
    data_loader = MNISTLoader()#数据读取
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)#优化损失函数

        #模型训练
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss,model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        #模型评估
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
    #pre_y = model.predict(data)
    #保存训练后的模型
    #tf.saved_model.save(model,"model")
    model.save("model")
    model1 = tf.keras.models.load_model("model")
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
    plot_model(model1, to_file="model.png", show_shapes=True, show_layer_names=True, rankdir='TB')
    plt.figure(figsize=(15, 15))
    img = plt.imread("model.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#pre()