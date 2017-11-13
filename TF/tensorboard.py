import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from TF_CNN import CNN


today = datetime.now().strftime('%y%m%d-%H%M')
tf.set_random_seed(777)

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

learning_rate = 0.001
batch_size = 100  
# mnist.train.num_examples는 55000이므로 55000/100 => 1 epoch 당 550번 반복
epochs = 1
# 원래 epochs = 15라 15 * 550이지만 속도 때문에 2 * 550으로 변경.

sess = tf.Session()
models = []

def tf_train_wrap(restore_step=0, ensemble=1, tflogs=True):
    ### Ensemble ###
    for m in range(ensemble):
        models.append(CNN(sess, "model" + str(m), learning_rate))
    
    saver = tf.train.Saver()
    if restore_step:
        saver.restore(sess, "./Variables/mnist-en{}.ckpt-{}".format(ensemble, restore_step))
    else:
        sess.run(tf.global_variables_initializer())
    
    step = 0
    for epoch in range(epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(mnist.train.num_examples / batch_size)
        total_batch = int(total_batch/10) #속도 때문에 추가. total_batch = 55
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)    
            for m_idx, m in enumerate(models):
                ### run Neural Net
                if tflogs:
                    writer = tf.summary.FileWriter('./tflogs/log-'+today, graph=sess.graph)
                    #writer.add_graph(sess.graph)
                    c, _, s = m.train(batch_xs, batch_ys)
                    writer.add_summary(s, global_step=step)
                else:
                    c, _ = m.train(batch_xs, batch_ys, summary=False)
                step += 1
                avg_cost_list[m_idx] += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
        save_path = saver.save(sess, "./Variables/mnist-en{}.ckpt-{}".format(ensemble, step+restore_step))
        

def tf_prediction_wrap():
    pass

def _check(sess, models):
    test_size = len(mnist.test.labels[:1000])
    predictions = np.zeros(test_size * 10).reshape(test_size, 10)
    
    for m_idx, m in enumerate(models):
        print(m_idx, "Accuracy   :", m.get_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        p = m.prediction(mnist.test.images[:1000])
        predictions += p 
        
    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(mnist.test.labels[:1000], axis=1))
    #캐스팅 한 다음 다 더해서 개수만큼 나누기 = 평균이니까 reduce_mean.
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
    print("Ensemble accuracy : ", sess.run(ensemble_accuracy))

if __name__ == '__main__':
    tf_train_wrap(restore_step=55)
